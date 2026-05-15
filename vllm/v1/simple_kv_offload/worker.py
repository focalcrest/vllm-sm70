# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

import mmap
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend = DmaCopyBackend()

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}

        # SM70 fork (Phase E.1): if cpu_pool_path is configured, CPU pool
        # tensors are backed by mmap'd files (NVMe for SSD spillover, or
        # /dev/shm for cross-process DP sharing in Phase E.2). Hold the mmap
        # handles here so the GC doesn't unmap them while the worker is live.
        self._mmap_handles: list[mmap.mmap] = []

        # SM70 fork (Phase E.5a): optional L2 tier (typically XFS on NVMe).
        # Only L1 participates in GPU<->CPU DMA; L2 acts as staging storage
        # and requires a promote (L2->L1 CPU memcpy) before its block becomes
        # DMA-eligible. Empty until register_kv_caches() runs with l2_*
        # extra_config keys set.
        self.l2_kv_caches: dict[str, torch.Tensor] | None = None
        self.num_l2_blocks: int = 0
        self._l2_mmap_handles: list[mmap.mmap] = []
        self._mmap_eng_id_short: str = ""

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.
        The worker will infer the underlying raw storage from the kv_caches.

        Args:
            kv_caches: Per-layer GPU KV caches. Values are either a single
                tensor (attention layers) or a list of tensors (Mamba layers
                in hybrid models). All values are included for offloading
                by resolving to their underlying raw storage.
        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        # Resolve each entry to a representative tensor for storage
        # deduplication. For attention layers the value is already a tensor;
        # for Mamba layers it is a list of tensors that all share the same
        # underlying raw storage, so we take the first one.
        def _repr_tensor(v: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
            assert isinstance(v, torch.Tensor | list)
            return v if isinstance(v, torch.Tensor) else v[0]

        any_tensor = _repr_tensor(next(iter(kv_caches.values())))
        self.device = any_tensor.device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks

        # Deduplicate: multiple layers may share the same backing storage.
        seen_ptrs: dict[int, tuple[str, torch.Tensor]] = {}
        for name, value in kv_caches.items():
            tensor = _repr_tensor(value)
            ptr = tensor.untyped_storage().data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = (name, tensor)

        # Build [num_blocks, block_bytes] int8 views from each unique
        # storage so that stride(0) gives block_bytes for the copy op.
        #
        # The physical layout varies across attention backends:
        #   FlashAttn/ROCm:  (2, num_blocks, ...) -> K/V outermost, 2 segments
        #   FlashInfer/MLA:  (num_blocks, ...)    -> blocks outermost, 1 segment
        # We derive page_size_bytes = storage.nbytes() // num_blocks, then
        # classify dims: any dim whose byte-stride exceeds page_size_bytes
        # must be an outer segment dim (e.g. the K/V dim of size 2). A less
        # hacky way is to update the interface with the layout.
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        for name, tensor in seen_ptrs.values():
            storage = tensor.untyped_storage()
            raw = torch.empty(0, dtype=torch.int8, device=self.device).set_(
                storage, 0, (storage.nbytes(),)
            )
            el = tensor.element_size()
            page_size_bytes = storage.nbytes() // num_blocks
            outer_dims = [
                d for d in range(tensor.ndim) if tensor.stride(d) * el > page_size_bytes
            ]
            if not outer_dims:
                unique_gpu_caches[name] = raw.view(num_blocks, -1)
            else:
                seg_stride = tensor.stride(outer_dims[0]) * el
                for idx in range(tensor.shape[outer_dims[0]]):
                    offset = idx * seg_stride
                    chunk = raw[offset : offset + seg_stride]
                    unique_gpu_caches[f"{name}.{idx}"] = chunk.view(num_blocks, -1)

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)

        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        logger.info(
            "SimpleCPUOffloadWorker: %d unique GPU KV tensors, "
            "allocating %d CPU blocks (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        # SM70 fork (Phase E.1): mmap-backed CPU pool for SSD spillover.
        # Set kv_connector_extra_config = {"cpu_pool_path": "/mnt/nvme/vllm_kv"}
        # or "/dev/shm/vllm_kv" to back the CPU tensors with mmap'd files.
        # When unset, falls back to torch.zeros pinned-memory allocation
        # (the upstream default).
        kv_xfer_cfg = self.vllm_config.kv_transfer_config
        extra_cfg = (
            kv_xfer_cfg.kv_connector_extra_config
            if kv_xfer_cfg is not None
            else {}
        ) or {}
        cpu_pool_path = extra_cfg.get("cpu_pool_path")
        # SM70 fork (Phase E.5a): optional L2 tier on a slower / larger
        # medium (e.g., XFS on NVMe). If both `l2_pool_path` and
        # `l2_bytes_to_use` are set, the worker allocates a second pool;
        # only L1 (the existing pool) participates in GPU<->CPU DMA, so
        # the manager must promote L2 blocks back to L1 before scheduling
        # any GPU transfer.
        l2_pool_path = extra_cfg.get("l2_pool_path")
        l2_bytes_to_use = int(extra_cfg.get("l2_bytes_to_use", 0) or 0)

        pin_memory = is_pin_memory_available()

        # SM70 fork (Phase E.2a): include engine_id in mmap path so DP=2
        # co-tenancy doesn't have two engines clobbering each other's
        # files. UUID4 prefix is unique enough; use first 8 hex chars.
        eng_id_full = (
            kv_xfer_cfg.engine_id
            if (kv_xfer_cfg and kv_xfer_cfg.engine_id)
            else "unknown"
        )
        self._mmap_eng_id_short = eng_id_full.replace("-", "")[:8]

        if not (cpu_pool_path or pin_memory):
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        self.gpu_kv_caches = unique_gpu_caches
        # Allocate L1 pool (existing path; renamed local var for clarity).
        self.cpu_kv_caches, self.num_cpu_blocks = self._allocate_pool(
            tier_label="l1",
            pool_path=cpu_pool_path,
            capacity_bytes=self.cpu_capacity_bytes,
            unique_gpu_caches=unique_gpu_caches,
            total_bytes_per_block=total_bytes_per_block,
            pin_memory=pin_memory,
            mmap_handles_target=self._mmap_handles,
        )

        # Allocate L2 pool if configured. L2 is mmap-only by design — its
        # whole point is SSD spillover. We refuse a pinned-host L2 because
        # the math doesn't work (would defeat the RAM budget).
        if l2_pool_path and l2_bytes_to_use > 0:
            self.l2_kv_caches, self.num_l2_blocks = self._allocate_pool(
                tier_label="l2",
                pool_path=l2_pool_path,
                capacity_bytes=l2_bytes_to_use,
                unique_gpu_caches=unique_gpu_caches,
                total_bytes_per_block=total_bytes_per_block,
                pin_memory=False,  # L2 must be mmap-backed; never pin
                mmap_handles_target=self._l2_mmap_handles,
            )
            logger.info(
                "SimpleCPUOffloadWorker: L2 tier allocated at %s, "
                "num_l2_blocks=%d (%.2f GB)",
                l2_pool_path,
                self.num_l2_blocks,
                (self.num_l2_blocks * total_bytes_per_block) / (1024**3),
            )

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        # Initialize copy backend with caches and streams.
        # Only L1 participates in GPU<->CPU DMA. L2 is staging storage that
        # the manager must promote into L1 before any DMA can reference it.
        self._backend.init(
            self.gpu_kv_caches,
            self.cpu_kv_caches,
            self.device,
            self.load_stream,
            self.store_stream,
        )

    def _allocate_pool(
        self,
        tier_label: str,
        pool_path: str | None,
        capacity_bytes: int,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        pin_memory: bool,
        mmap_handles_target: list[mmap.mmap],
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Allocate one tier of CPU KV mirror tensors.

        SM70 fork (Phase E.5a): factored out of register_kv_caches so we
        can call it once for L1 (`/dev/shm`) and once for L2 (`/kvcache`).

        ``tier_label`` ("l1" / "l2") becomes part of the mmap file name so
        the two tiers don't collide on disk when sharing a directory.

        Returns ``(cpu_caches_dict, num_blocks)``.
        """
        num_blocks = max(1, capacity_bytes // total_bytes_per_block)
        use_mmap = bool(pool_path)
        if use_mmap:
            os.makedirs(pool_path, exist_ok=True)
            rank = self.device.index if self.device is not None else 0
            logger.info(
                "SimpleCPUOffloadWorker: %s mmap pool at %s "
                "(engine=%s rank=%d, num_blocks=%d)",
                tier_label.upper(), pool_path,
                self._mmap_eng_id_short, rank, num_blocks,
            )

        cpu_caches: dict[str, torch.Tensor] = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (num_blocks,) + gpu_tensor.shape[1:]
            if use_mmap:
                safe_name = name.replace("/", "_").replace(".", "_")
                rank = self.device.index if self.device is not None else 0
                file_path = os.path.join(
                    pool_path,
                    f"eng{self._mmap_eng_id_short}_r{rank}_"
                    f"{tier_label}_{safe_name}.bin",
                )
                total_bytes = (
                    int(np.prod(cpu_shape)) * gpu_tensor.element_size()
                )
                fd = os.open(file_path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    os.ftruncate(fd, total_bytes)
                    mm = mmap.mmap(
                        fd, total_bytes, mmap.MAP_SHARED,
                        mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                finally:
                    os.close(fd)
                mmap_handles_target.append(mm)
                np_dtype_str = {
                    torch.float16: "f2",
                    torch.bfloat16: "f2",
                    torch.float32: "f4",
                    torch.uint8: "u1",
                    torch.int8: "i1",
                    torch.int32: "i4",
                    torch.int64: "i8",
                }.get(gpu_tensor.dtype)
                if np_dtype_str is None:
                    raise NotImplementedError(
                        f"mmap CPU pool: unsupported dtype {gpu_tensor.dtype}"
                    )
                if gpu_tensor.dtype == torch.bfloat16:
                    arr = np.frombuffer(mm, dtype=np.uint16).reshape(cpu_shape)
                    tensor = torch.from_numpy(arr).view(torch.bfloat16)
                else:
                    arr = np.frombuffer(mm, dtype=np_dtype_str).reshape(cpu_shape)
                    tensor = torch.from_numpy(arr)
            else:
                tensor = torch.zeros(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
                if pin_memory:
                    pin_tensor(tensor)
            cpu_caches[name] = tensor
        return cpu_caches, num_blocks

    def cpu_tier_move(
        self,
        pairs: list,
        src_tier: str,
        dst_tier: str,
    ) -> None:
        """Copy whole blocks between L1 and L2 via CPU memcpy.

        SM70 fork (Phase E.5a). Used by the scheduler's demote/promote
        orchestration in Phase E.5b. ``pairs`` is a list of
        ``(src_block_id, dst_block_id)`` tuples. Synchronous; runs on the
        calling (worker) thread.

        Tier labels: ``"l1"`` -> ``self.cpu_kv_caches``,
        ``"l2"`` -> ``self.l2_kv_caches``. Same-tier moves are a no-op.
        """
        if not pairs or src_tier == dst_tier:
            return
        if src_tier == "l1":
            src = self.cpu_kv_caches
        elif src_tier == "l2":
            src = self.l2_kv_caches
        else:
            raise ValueError(f"Unknown src_tier: {src_tier!r}")
        if dst_tier == "l1":
            dst = self.cpu_kv_caches
        elif dst_tier == "l2":
            dst = self.l2_kv_caches
        else:
            raise ValueError(f"Unknown dst_tier: {dst_tier!r}")
        if src is None or dst is None:
            raise RuntimeError(
                f"cpu_tier_move({src_tier}->{dst_tier}) requested but one "
                f"of the tiers is not allocated. Did you set l2_pool_path?"
            )
        for src_id, dst_id in pairs:
            for name, src_t in src.items():
                dst[name][dst_id].copy_(src_t[src_id])

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # NOTE: we defer launching both load and store to get_finished(),
        # which runs after model execution. This hides the CPU-side
        # block copy op overhead (~5ms) behind GPU compute.
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Submit transfers and report completed events to the scheduler.

        Called after model execution. The manager only schedules stores for
        blocks whose KV data is confirmed computed, so we launch both loads
        and stores immediately — no deferral or cross-stream sync needed.

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.
        """
        # (1) Submit transfers
        metadata = self._connector_metadata
        if metadata is not None:
            # SM70 fork (Phase E.5a): inter-tier moves run BEFORE any DMA.
            # Promotes (L2->L1) populate the L1 slots that this step's load
            # DMA will copy to GPU. Demotes (L1->L2) preserve L1-resident
            # data before this step's store DMA overwrites the slot.
            # Synchronous CPU memcpy on the worker thread; cost is bounded
            # by len(pairs) * block_bytes (~15-30 ms per 50 MB block).
            if getattr(metadata, "promote_pairs", None):
                self.cpu_tier_move(
                    metadata.promote_pairs, src_tier="l2", dst_tier="l1"
                )
            if getattr(metadata, "demote_pairs", None):
                self.cpu_tier_move(
                    metadata.demote_pairs, src_tier="l1", dst_tier="l2"
                )

            # Launch loads (CPU->GPU).
            if metadata.load_cpu_blocks:
                self._backend.launch_copy(
                    metadata.load_cpu_blocks,
                    metadata.load_gpu_blocks,
                    is_store=False,
                    event_idx=metadata.load_event,
                    events_list=self._load_events,
                )
            # Launch stores (GPU->CPU).
            if metadata.store_gpu_blocks:
                self._backend.launch_copy(
                    metadata.store_gpu_blocks,
                    metadata.store_cpu_blocks,
                    is_store=True,
                    event_idx=metadata.store_event,
                    events_list=self._store_events,
                )

        # (2) Track completed transfer events
        finished_recving: set[str] = set()

        if self._pending_load_event_indices:
            load_wm = self._poll_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(j)
                req_ids = (
                    metadata.load_event_to_reqs.get(j) if metadata is not None else None
                )
                if req_ids:
                    finished_recving.update(req_ids)

        if self._pending_store_event_indices:
            store_wm = self._poll_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                self._completed_store_events[j] = 1

        return None, finished_recving or None

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return completed store events since the last call."""
        if not self._completed_store_events:
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta

    def handle_preemptions(
        self, kv_connector_metadata: SimpleCPUOffloadMetadata
    ) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        if not kv_connector_metadata.need_flush:
            return
        self._flush_and_sync_all()

    def _flush_and_sync_all(self) -> None:
        """Synchronize all in-flight transfer events."""
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()

        for event_idx, event in self._store_events:
            event.synchronize()
            self._store_hwm = event_idx
        self._store_events.clear()

    def _poll_stream_events(self, is_store: bool) -> int:
        """Non-blocking poll for completed events and return the high-water mark."""
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm
