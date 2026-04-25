# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    gpu_p2p_access_check,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False

logger = init_logger(__name__)


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if envs.VLLM_SKIP_P2P_CHECK:
            logger.debug("Skipping P2P check and trusting the driver's P2P report.")
            return torch.cuda.can_device_access_peer(rank, i)
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size=8192 * 1024,
        symm_mem_enabled=False,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bound to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True
        self.hierarchical = False

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            logger.info(
                "Custom allreduce is disabled because "
                "of missing custom allreduce library"
            )
            return

        self.group = group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group."
        )

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes."
            )
            return

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        device_capability = current_platform.get_device_capability()
        if (
            current_platform.is_cuda()
            and symm_mem_enabled
            and device_capability is not None
        ):
            device_capability_str = device_capability.as_version_str()
            if device_capability_str in CUSTOM_ALL_REDUCE_MAX_SIZES:
                max_size = min(
                    CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str][world_size],
                    max_size,
                )
        cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(current_platform.device_count()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert current_platform.is_cuda_alike()
        fully_connected = current_platform.is_fully_connected(physical_device_ids)
        hierarchical = False
        groups: list[list[int]] = []
        if world_size > 2 and not fully_connected:
            groups = CustomAllreduce._detect_nvlink_groups(physical_device_ids)
            if len(groups) == 2 and len(groups[0]) == len(groups[1]):
                has_partners = all(
                    CustomAllreduce._find_nvlink_partner(
                        r, physical_device_ids, groups
                    )
                    is not None
                    for r in range(world_size)
                )
                if has_partners:
                    hierarchical = True
                    logger.info(
                        "Hierarchical custom allreduce enabled: %s with "
                        "cross-group NVLink partners",
                        groups,
                    )
        if world_size > 2 and not fully_connected and not hierarchical:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
            logger.warning(
                "Custom allreduce is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly."
            )
            return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Metadata composes of two parts: metadata for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(
            ops.meta_size() + max_size, group=group, uncached=True
        )
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected
        self.hierarchical = hierarchical
        if hierarchical:
            group_id = 0 if rank in groups[0] else 1
            local_rank = groups[group_id].index(rank)
            partner = CustomAllreduce._find_nvlink_partner(
                rank, physical_device_ids, groups
            )
            assert partner is not None
            self._ptr = ops.init_custom_ar_hierarchical(
                self.meta_ptrs,
                self.rank_data,
                rank,
                group_id,
                local_rank,
                partner,
            )
            ops.register_buffer(self._ptr, self.buffer_ptrs)
            ops.register_group_buffer(self._ptr, self.buffer_ptrs)
        else:
            self._ptr = ops.init_custom_ar(
                self.meta_ptrs, self.rank_data, rank, self.fully_connected
            )
            ops.register_buffer(self._ptr, self.buffer_ptrs)

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data: list[list[list[int] | None]]
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )
        # Unpack list of tuples to tuple of lists.
        handles = cast(list[list[int]], [d[0] for d in all_data])
        offsets = cast(list[list[int]], [d[1] for d in all_data])
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.max_size
        if self.hierarchical:
            return inp_size < self.max_size
        return False

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    @staticmethod
    def _detect_nvlink_groups(
        physical_device_ids: list[int],
    ) -> list[list[int]]:
        """Use NVML to detect NVLink connectivity and cluster GPUs into
        fully-connected groups. For V100-SXM2 returns [[0,1,2,3], [4,5,6,7]]."""
        import pynvml

        pynvml.nvmlInit()
        n = len(physical_device_ids)

        # Build PCI bus ID → rank mapping
        pci_bus_to_rank: dict[str, int] = {}
        for rank, dev_id in enumerate(physical_device_ids):
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_bus_to_rank[pci.busId] = rank

        # Find NVLink connections for each device
        nvlink_adj: list[set[int]] = [set() for _ in range(n)]
        for rank, dev_id in enumerate(physical_device_ids):
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            for link in range(6):
                try:
                    if pynvml.nvmlDeviceGetNvLinkState(handle, link):
                        remote_pci = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(
                            handle, link
                        )
                        remote_bus = remote_pci.busId
                        if remote_bus in pci_bus_to_rank:
                            nvlink_adj[rank].add(
                                pci_bus_to_rank[remote_bus]
                            )
                except pynvml.NVMLError:
                    continue
        pynvml.nvmlShutdown()

        # Cluster into fully-connected groups (greedy clique finding)
        visited = [False] * n
        groups: list[list[int]] = []
        for i in range(n):
            if visited[i]:
                continue
            group = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j] and all(
                    j in nvlink_adj[k] and k in nvlink_adj[j]
                    for k in group
                ):
                    group.append(j)
                    visited[j] = True
            groups.append(group)
        return groups

    @staticmethod
    def _find_nvlink_partner(
        rank: int,
        physical_device_ids: list[int],
        groups: list[list[int]],
    ) -> int | None:
        """Find cross-group NVLink partner for a given rank.
        V100: 0↔4, 1↔5, 2↔6, 3↔7"""
        import pynvml

        pynvml.nvmlInit()
        my_phys = physical_device_ids[rank]
        handle = pynvml.nvmlDeviceGetHandleByIndex(my_phys)

        # Build PCI bus ID → rank mapping
        pci_bus_to_rank: dict[str, int] = {}
        for r, dev_id in enumerate(physical_device_ids):
            h = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            pci_bus_to_rank[pynvml.nvmlDeviceGetPciInfo(h).busId] = r

        my_group_idx = next(
            i for i, g in enumerate(groups) if rank in g
        )
        partner = None
        for link in range(6):
            try:
                if pynvml.nvmlDeviceGetNvLinkState(handle, link):
                    remote_pci = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(
                        handle, link
                    )
                    remote_rank = pci_bus_to_rank.get(remote_pci.busId)
                    if remote_rank is not None and remote_rank != rank:
                        other_group_idx = next(
                            i
                            for i, g in enumerate(groups)
                            if remote_rank in g
                        )
                        if other_group_idx != my_group_idx:
                            partner = remote_rank
                            break
            except pynvml.NVMLError:
                continue
        pynvml.nvmlShutdown()
        return partner

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, registered=False)

    def close(self):
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: ProcessGroup | None = None,
        uncached: bool | None = False,
    ) -> list[int]:
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: list[int],
        group: ProcessGroup | None = None,
        rank: int | None = None,
    ) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])
