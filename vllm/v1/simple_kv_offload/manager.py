# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

import contextlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_utils import get_block_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class MarconiAdmissionTracker:
    """SM70 fork (Phase E.3a): Marconi-style admission filter.

    Tracks how often each block_hash has been observed across requests.
    `should_admit(bhash)` returns True only if the hash has been seen at
    least `threshold` times — i.e. we predict the prefix will be reused.

    Default threshold=1 reproduces the upstream behavior ("admit
    everything"). Set via kv_connector_extra_config["admission_threshold"]
    (e.g. 2: only cache prefixes that have appeared twice).

    The hash_count dict is GC'd when it exceeds max_size: drops all
    singletons (seen exactly once), keeping the "hot" working set.
    """

    def __init__(self, threshold: int = 1, max_size: int = 100_000):
        self.threshold: int = max(1, int(threshold))
        self.max_size: int = max_size
        self.hash_count: dict[bytes, int] = {}
        # Cheap stats for worklog/benchmarking.
        self.admitted: int = 0
        self.rejected: int = 0
        self.observed: int = 0

    def observe(self, bhash: bytes) -> None:
        if bhash is None:
            return
        self.observed += 1
        self.hash_count[bhash] = self.hash_count.get(bhash, 0) + 1
        if len(self.hash_count) > self.max_size:
            # GC: drop singletons to bound memory; preserves "hot" set.
            self.hash_count = {
                h: c for h, c in self.hash_count.items() if c >= 2
            }

    def should_admit(self, bhash: bytes) -> bool:
        if self.threshold <= 1:
            self.admitted += 1
            return True
        admit = self.hash_count.get(bhash, 0) >= self.threshold
        if admit:
            self.admitted += 1
        else:
            self.rejected += 1
        return admit

    def stats(self) -> dict[str, int]:
        return {
            "threshold": self.threshold,
            "observed": self.observed,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "tracked_hashes": len(self.hash_count),
            "hot_hashes": sum(
                1 for c in self.hash_count.values() if c >= self.threshold
            ),
        }


class MarconiEvictionPolicy:
    """SM70 fork (Phase E.3b): reuse-aware "gateway" eviction policy.

    Pairs with [[MarconiAdmissionTracker]]. Marconi (NSDI'25) was designed
    for hybrid layouts where deep blocks store the most compute. vLLM
    prefix caching is strictly left-to-right: a depth-d block is useless
    without depths 0..d-1, so evicting any shallow block of a hot prefix
    kills the whole chain. That makes the textbook FLOP-aware scoring
    actively harmful — our Zipfian eval showed it lost LRU by ~115 ms
    mean. We invert the depth term to prefer keeping *gateway* blocks of
    hot prefixes:

        score = reuse_count(bhash) * exp(-decay_rate * age) / (depth + 1)

    Where ``reuse_count`` is sourced from the admission tracker's
    observed-hash histogram (already counted by ``observe`` on every
    `get_num_new_matched_tokens` call). Lower-scored blocks are evicted
    first, so:
      - cold prefixes (low reuse_count) evict before hot ones
      - within a hot prefix, deep blocks evict before shallow ones
        (preserves chain integrity)
      - long-idle blocks decay out gradually

    Default decay_rate=0.0 keeps pure reuse-and-depth ranking.
    """

    def __init__(self, decay_rate: float = 0.0, reuse_mode: str = "linear"):
        # SM70 fork (Phase E.5b): metadata is keyed by (tier, block_id).
        # Tier is a string ("l1" / "l2") so L1/L2 block_ids don't collide.
        # Existing single-tier callers omit the tier arg; the default "l1"
        # preserves the previous behavior.
        self.metadata: dict[
            tuple[str, int], tuple[int, int, bytes | None]
        ] = {}
        self.step: int = 0
        self.decay_rate: float = float(decay_rate)
        # "linear" -> reuse_count (concentrates cache on the single hottest
        # prefix); "log" -> log(1 + reuse_count) (balances hot vs warm,
        # recommended for Zipfian workloads); "none" -> ignore reuse, pure
        # gateway preservation.
        self.reuse_mode: str = str(reuse_mode).lower()
        self.evictions: int = 0
        self.hoisted: int = 0
        # Tier-counter for Phase E.5b orchestration diagnostics.
        self.demotes: int = 0
        self.promotes: int = 0
        # Callable[[bytes], int] returning observed reuse count for a hash.
        # Wired in by the scheduler so we don't keep a hard reference loop.
        self.reuse_count_fn = None

    def bind_reuse_lookup(self, fn) -> None:
        self.reuse_count_fn = fn

    def _reuse_factor(self, bhash: bytes | None) -> float:
        if self.reuse_mode == "none" or bhash is None or self.reuse_count_fn is None:
            return 1.0
        try:
            raw = max(1, int(self.reuse_count_fn(bhash)))
        except Exception:
            return 1.0
        if self.reuse_mode == "log":
            return math.log1p(raw)
        # default: "linear"
        return float(raw)

    def tick(self) -> None:
        self.step += 1

    def record_store(
        self,
        cpu_block_id: int,
        depth: int,
        bhash: bytes | None = None,
        tier: str = "l1",
    ) -> None:
        self.metadata[(tier, cpu_block_id)] = (int(depth), self.step, bhash)

    def record_access(self, cpu_block_id: int, tier: str = "l1") -> None:
        key = (tier, cpu_block_id)
        meta = self.metadata.get(key)
        if meta is not None:
            depth, _, bhash = meta
            self.metadata[key] = (depth, self.step, bhash)

    def forget(self, cpu_block_id: int, tier: str = "l1") -> None:
        self.metadata.pop((tier, cpu_block_id), None)

    def transfer_tier(
        self, src_tier: str, src_id: int, dst_tier: str, dst_id: int
    ) -> None:
        """SM70 fork (E.5b): move metadata across tiers on demote/promote.

        Keeps depth, last_step, bhash intact; only the tier+block_id key
        changes. If src has no entry (block was untracked, e.g., evicted
        before we got here), this is a no-op.
        """
        src_key = (src_tier, src_id)
        meta = self.metadata.pop(src_key, None)
        if meta is None:
            return
        self.metadata[(dst_tier, dst_id)] = meta

    def score(self, cpu_block_id: int, tier: str = "l1") -> float:
        meta = self.metadata.get((tier, cpu_block_id))
        if meta is None:
            # Unknown block -> evict first.
            return -1.0
        depth, last_step, bhash = meta
        base = self._reuse_factor(bhash) / (depth + 1.0)
        if self.decay_rate <= 0.0:
            return base
        age = self.step - last_step
        return base * math.exp(-self.decay_rate * age)

    def stats(self) -> dict[str, float | int]:
        if self.metadata:
            avg_depth = sum(d for d, _, _ in self.metadata.values()) / len(
                self.metadata
            )
        else:
            avg_depth = 0.0
        avg_reuse = 0.0
        if self.reuse_count_fn is not None and self.metadata:
            total = 0
            n = 0
            for _, _, bhash in self.metadata.values():
                if bhash is None:
                    continue
                try:
                    total += max(1, int(self.reuse_count_fn(bhash)))
                    n += 1
                except Exception:
                    continue
            avg_reuse = total / n if n else 0.0
        n_l1 = sum(1 for (t, _) in self.metadata if t == "l1")
        n_l2 = sum(1 for (t, _) in self.metadata if t == "l2")
        return {
            "step": self.step,
            "tracked": len(self.metadata),
            "l1": n_l1,
            "l2": n_l2,
            "evictions": self.evictions,
            "hoisted": self.hoisted,
            "demotes": self.demotes,
            "promotes": self.promotes,
            "avg_tracked_depth": round(avg_depth, 2),
            "avg_tracked_reuse": round(avg_reuse, 2),
        }


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]


@dataclass
class LoadRequestState:
    request: "Request"
    transfer_meta: TransferMeta
    load_event: int | None = None
    finished: bool = False


# NOTE: This per-request state is only used in eager mode.
@dataclass
class StoreRequestState:
    request: "Request"
    # Accumulated block IDs from scheduler_output via yield_req_data.
    block_ids: tuple[list[int], ...]
    # Per-group cursors tracking how many blocks have been stored/skipped.
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)
    finished: bool = False


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        lazy_offload: bool = False,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        # NOTE: We use the same block size for both GPU and CPU.
        self.block_size = vllm_config.cache_config.block_size
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        assert kv_cache_config is not None
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, cpu_capacity_bytes
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks
        # Find the full attention kv group for prefix cache matching.
        self.fa_gidx = -1
        for g_idx, g in enumerate(self.cpu_kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                self.fa_gidx = g_idx
                break
        assert 0 <= self.fa_gidx < len(self.cpu_kv_cache_config.kv_cache_groups)

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d CPU blocks (%.2f GB, mode=%s)",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
        )

        # TODO (yifan): maybe need to enable kv_cache_events and metrics_collector here.
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        assert dcp_world_size == 1 and pcp_world_size == 1
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=self.block_size,
        )
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool

        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # SM70 fork: hit-rate instrumentation for the CPU offload path.
        # Counters are incremented inside get_num_new_matched_tokens; a
        # cumulative summary is logged every _hit_log_every requests so
        # external tooling can grep the log.
        self._req_seen: int = 0
        self._req_hits: int = 0
        self._hit_tokens_total: int = 0
        self._req_token_total: int = 0
        self._hit_log_every: int = 10

        # Load metadata
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        # Lazy mode: use a cursor to track the last scanned block in the GPU free queue.
        self._cursor: KVCacheBlock | None = None
        if self._lazy_mode:
            self._target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
        else:
            self._target_free = 0
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}

        # SM70 fork (Phase E.3a): Marconi-style admission tracker.
        # extra_config["admission_threshold"]=N requires a block_hash to have
        # been observed >=N times before it gets stored to CPU pool.
        kv_xfer_cfg = vllm_config.kv_transfer_config
        extra_cfg = (
            kv_xfer_cfg.kv_connector_extra_config
            if kv_xfer_cfg is not None
            else {}
        ) or {}
        admission_threshold = int(extra_cfg.get("admission_threshold", 1))
        self._admission_tracker = MarconiAdmissionTracker(
            threshold=admission_threshold,
        )
        if admission_threshold > 1:
            logger.info(
                "SimpleCPUOffloadScheduler: Marconi admission enabled, "
                "threshold=%d",
                admission_threshold,
            )

        # SM70 fork (Phase E.3b): FLOP-aware eviction policy.
        # extra_config["eviction_policy"] = "marconi" enables score-based
        # eviction; default "lru" preserves the upstream behavior.
        # extra_config["eviction_decay"] = float blends recency into score.
        # SM70 fork (Phase E.5b): optional L2 tier (XFS on NVMe). When
        # ``l2_pool_path`` and ``l2_bytes_to_use`` are set in extra_config
        # (must match the worker's settings), we build a separate
        # BlockPool/Coordinator for the L2 tier and run a tier-aware
        # admit/promote/demote orchestration on top of the existing
        # single-tier code paths.
        l2_capacity_bytes = int(extra_cfg.get("l2_bytes_to_use", 0) or 0)
        l2_pool_path_present = bool(extra_cfg.get("l2_pool_path"))
        self._dual_tier: bool = l2_capacity_bytes > 0 and l2_pool_path_present
        self.l2_kv_cache_config: KVCacheConfig | None = None
        self.l2_coordinator: KVCacheCoordinator | None = None
        self.l2_block_pool: BlockPool | None = None
        self.num_l2_blocks: int = 0
        if self._dual_tier:
            self.l2_kv_cache_config = self._derive_cpu_config(
                kv_cache_config, l2_capacity_bytes
            )
            self.num_l2_blocks = self.l2_kv_cache_config.num_blocks
            self.l2_coordinator = get_kv_cache_coordinator(
                kv_cache_config=self.l2_kv_cache_config,
                max_model_len=vllm_config.model_config.max_model_len,
                use_eagle=False,
                enable_caching=True,
                enable_kv_cache_events=False,  # L2 is internal; no events
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                hash_block_size=self.block_size,
            )
            self.l2_block_pool = self.l2_coordinator.block_pool
            logger.info(
                "SimpleCPUOffloadScheduler: L2 tier enabled, "
                "num_l2_blocks=%d (%.2f GB)",
                self.num_l2_blocks,
                l2_capacity_bytes / (1024**3),
            )

        # SM70 fork (Phase E.5b): hit-rate split across tiers + miss.
        # Updated alongside the existing _req_seen / _req_hits / _req_token
        # counters whenever get_num_new_matched_tokens decides.
        self._req_l1_hits: int = 0
        self._req_l2_hits: int = 0
        self._tok_l1_hits: int = 0
        self._tok_l2_hits: int = 0

        # SM70 fork (Phase E.5b): demote/promote pair accumulators. These
        # are populated by _prepare_eager_store_specs (demote) and
        # update_state_after_alloc (promote), then drained by
        # build_connector_meta into SimpleCPUOffloadMetadata so the worker
        # can execute the corresponding CPU memcpys before any DMA.
        self._pending_demote_pairs: list[tuple[int, int]] = []
        self._pending_promote_pairs: list[tuple[int, int]] = []
        # SM70 fork (Phase E.5b-2): per-request tier choice made by
        # get_num_new_matched_tokens, consumed by update_state_after_alloc.
        # Values: "l1" or "l2" (only set when hit_length > 0).
        self._pending_hit_tier: dict[str, str] = {}

        eviction_policy_name = str(
            extra_cfg.get("eviction_policy", "lru")
        ).lower()
        eviction_decay = float(extra_cfg.get("eviction_decay", 0.0))
        eviction_reuse_mode = str(
            extra_cfg.get("eviction_reuse_mode", "linear")
        ).lower()
        self._eviction_policy: MarconiEvictionPolicy | None = None
        if eviction_policy_name == "marconi":
            self._eviction_policy = MarconiEvictionPolicy(
                decay_rate=eviction_decay,
                reuse_mode=eviction_reuse_mode,
            )
            # Reuse-count comes from the admission tracker's hash histogram.
            self._eviction_policy.bind_reuse_lookup(
                lambda bh, t=self._admission_tracker: t.hash_count.get(bh, 0)
            )
            logger.info(
                "SimpleCPUOffloadScheduler: Marconi eviction enabled, "
                "decay=%g, reuse_mode=%s",
                eviction_decay,
                eviction_reuse_mode,
            )

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig", cpu_capacity_bytes: int
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio."""
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfigCls
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert len(gpu_config.kv_cache_tensors) > 0

        gpu_total_bytes = sum(t.size for t in gpu_config.kv_cache_tensors)
        num_gpu_blocks = gpu_config.num_blocks
        num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)
        # Create CPU kv_cache_tensors mirroring GPU by scaling size proportionally.
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                shared_by=list(t.shared_by),
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return KVCacheConfigCls(
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
            kv_cache_groups=gpu_config.kv_cache_groups,
        )

    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: "KVCacheConfig", max_num_batched_tokens: int
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode."""
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.kv_cache_groups:
            spec = g.kv_cache_spec
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, spec.block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, spec.block_size)
        return int(target * (1 + WATERMARK_RATIO))

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""
        skipped = num_computed_tokens // self.block_size
        remaining_hashes = request.block_hashes[skipped:]

        # SM70 fork (Phase E.3a): Marconi admission observation. Every block
        # the scheduler asks about gets counted; the admission gate in
        # _prepare_eager_store_specs uses these counts to decide whether
        # a block is "hot enough" to deserve a CPU pool slot.
        for bh in remaining_hashes:
            self._admission_tracker.observe(bh)

        # SM70 fork: per-request hit-rate tracking. We measure prospective
        # CPU hit length on the *uncovered* suffix (after the GPU prefix
        # cache has done its work). hit_length=0 means "GPU prefix did the
        # work or pure recompute"; hit_length>0 means CPU pool saved this
        # request some prefill compute.
        self._req_seen += 1
        eligible_tokens = max(0, request.num_tokens - 1 - num_computed_tokens)
        self._req_token_total += eligible_tokens

        if not remaining_hashes:
            self._maybe_log_hit_stats()
            return 0, False
        # Must recompute at least the last token, matching the logic in
        # kv_cache_manager.get_computed_blocks().
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        if max_hit_len <= 0:
            self._maybe_log_hit_stats()
            return 0, False

        # SM70 fork (Phase E.5b-2): tier-aware find. Try L1 first, then L2;
        # pick whichever yields the longer consecutive-prefix hit. This
        # cannot resolve TRULY mixed-tier chains (e.g., depth-0 in L1 plus
        # depth-1..N in L2); those degrade to "L1 alone" length here.
        # Promotion-on-load below moves L2-resident hits into L1 before DMA.
        l1_blocks, l1_hit = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )
        chosen_tier = "l1"
        hit_length = l1_hit
        if self._dual_tier and self.l2_coordinator is not None and l1_hit < max_hit_len:
            _, l2_hit = self.l2_coordinator.find_longest_cache_hit(
                remaining_hashes, max_hit_len
            )
            if l2_hit > l1_hit:
                chosen_tier = "l2"
                hit_length = l2_hit

        if hit_length > 0:
            self._req_hits += 1
            self._hit_tokens_total += hit_length
            self._pending_hit_tier[request.request_id] = chosen_tier
            if chosen_tier == "l1":
                self._req_l1_hits += 1
                self._tok_l1_hits += hit_length
            else:
                self._req_l2_hits += 1
                self._tok_l2_hits += hit_length
            self._maybe_log_hit_stats()
            return hit_length, True
        self._maybe_log_hit_stats()
        return 0, False

    def _maybe_log_hit_stats(self) -> None:
        """Emit a cumulative CPU-pool hit-rate summary every N requests."""
        if self._req_seen == 0 or self._req_seen % self._hit_log_every != 0:
            return
        req_rate = self._req_hits / self._req_seen
        tok_rate = (
            self._hit_tokens_total / self._req_token_total
            if self._req_token_total > 0
            else 0.0
        )
        if self._dual_tier:
            logger.info(
                "[CPU-OFFLOAD HIT] seen=%d hits=%d req_hit_rate=%.3f "
                "tok_hit_rate=%.3f hit_tok_sum=%d elig_tok_sum=%d "
                "l1_req=%d l2_req=%d l1_tok=%d l2_tok=%d",
                self._req_seen,
                self._req_hits,
                req_rate,
                tok_rate,
                self._hit_tokens_total,
                self._req_token_total,
                self._req_l1_hits,
                self._req_l2_hits,
                self._tok_l1_hits,
                self._tok_l2_hits,
            )
            return
        logger.info(
            "[CPU-OFFLOAD HIT] seen=%d hits=%d req_hit_rate=%.3f "
            "tok_hit_rate=%.3f hit_tok_sum=%d elig_tok_sum=%d",
            self._req_seen,
            self._req_hits,
            req_rate,
            tok_rate,
            self._hit_tokens_total,
            self._req_token_total,
        )

    # TODO(yifan): this API now only matches the suffix part of the prefix cache. A more
    # general API should scan blocks in both GPU and CPU block pool in a single pass.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)

        # Store tracking (eager mode only). Register the request;
        # block IDs are accumulated from scheduler_output in
        # _prepare_eager_store_specs via yield_req_data.
        if not self._lazy_mode and req_id not in self._reqs_to_store:
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple([] for _ in range(num_groups)),
                num_stored_blocks=[0] * num_groups,
            )

        # SM70 fork (Phase E.5b-2): pop the tier choice that
        # get_num_new_matched_tokens recorded for this request. Default to
        # "l1" for safety (covers external callers that bypassed the
        # connector's gate, though that path is unusual).
        hit_tier = self._pending_hit_tier.pop(req_id, "l1")

        if num_external_tokens == 0:
            return

        num_blocks_to_load = num_external_tokens // self.block_size
        assert num_blocks_to_load > 0

        skipped = sum(blk.block_hash is not None for blk in blocks.blocks[self.fa_gidx])
        num_computed_tokens = skipped * self.block_size
        hashes_to_load = request.block_hashes[skipped : skipped + num_blocks_to_load]

        # Find CPU cached blocks across all groups in the chosen tier.
        max_hit_len = len(hashes_to_load) * self.block_size
        source_coordinator = (
            self.l2_coordinator
            if hit_tier == "l2" and self.l2_coordinator is not None
            else self.cpu_coordinator
        )
        cpu_hit_blocks, hit_length = source_coordinator.find_longest_cache_hit(
            hashes_to_load, max_hit_len
        )
        assert hit_length == num_external_tokens, (
            f"Expected {num_external_tokens} hit tokens, got {hit_length} "
            f"(tier={hit_tier})"
        )

        # Build transfer pairs across all groups.
        total_computed_tokens = num_computed_tokens + num_external_tokens
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []
        # SM70 fork (Phase E.5b-2): blocks freshly promoted from L2 already
        # carry ref_cnt=1 from get_new_blocks(), so they must NOT go through
        # touch() a second time (which would leak a ref).
        cpu_blocks_already_held: list[KVCacheBlock] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            # Number of blocks in the computed range for this group.
            g_block_size = kv_cache_groups[g].kv_cache_spec.block_size
            n_computed_g = cdiv(total_computed_tokens, g_block_size)

            # Back-trace: ext blocks sit at the tail of the computed range.
            gpu_ext_start = n_computed_g - n_ext_g
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                if cpu_blk.is_null:
                    continue
                # SM70 fork (Phase E.5b-2): promote L2-resident hits into
                # L1 before DMA. The promote routine returns an L1 block
                # with ref_cnt already bumped (get_new_blocks), so we
                # skip the touch() bookkeeping for those entries.
                already_touched = False
                if hit_tier == "l2":
                    promoted = self._promote_l2_block_to_l1(cpu_blk)
                    if promoted is None:
                        # No L1 slot available — can't honor this hit;
                        # skip the block (caller will recompute).
                        continue
                    cpu_blk = promoted
                    already_touched = True
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])
                cpu_block_ids.append(cpu_blk.block_id)
                if already_touched:
                    cpu_blocks_already_held.append(cpu_blk)
                else:
                    cpu_blocks_to_touch.append(cpu_blk)

        # Touch L1-resident hit blocks (ref_cnt 0 -> 1). Promoted blocks
        # already came back from get_new_blocks at ref_cnt=1, so we don't
        # double-bump them.
        self.cpu_block_pool.touch(cpu_blocks_to_touch)

        # SM70 fork (Phase E.3b): record CPU-block accesses for the eviction
        # policy so deep/hot blocks see their recency refreshed on each hit.
        if self._eviction_policy is not None:
            for blk in cpu_blocks_to_touch:
                self._eviction_policy.record_access(blk.block_id)
            for blk in cpu_blocks_already_held:
                # Promoted blocks are now L1-resident; bookkeep recency.
                self._eviction_policy.record_access(
                    blk.block_id, tier="l1"
                )

        # Touch GPU blocks to prevent freeing during async load
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.touch(
            [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
        )

        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request, transfer_meta=TransferMeta(gpu_block_ids, cpu_block_ids)
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        # --- Stores ---
        store_event = -1
        store_gpu, store_cpu, store_req_ids = self.prepare_store_specs(scheduler_output)
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu, store_cpu
            )
            if store_req_ids:  # For eager mode only, track req->blocks mapping
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    store_state = self._reqs_to_store.get(req_id)
                    if store_state is not None:
                        store_state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        for req_id, load_state in self._reqs_to_load.items():
            if load_state.load_event is not None:
                continue
            assert load_state.transfer_meta is not None
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event
            self._load_event_to_reqs[load_event] = load_req_ids

        # SM70 fork (Phase E.5b): drain accumulated inter-tier moves into
        # the worker metadata so the worker performs the CPU memcpys at
        # the start of get_finished, before any DMA dispatch.
        demote_pairs = self._pending_demote_pairs
        promote_pairs = self._pending_promote_pairs
        self._pending_demote_pairs = []
        self._pending_promote_pairs = []

        result = SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            load_event_to_reqs=self._load_event_to_reqs,
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
            demote_pairs=demote_pairs,
            promote_pairs=promote_pairs,
            need_flush=bool(scheduler_output.preempted_req_ids),
        )
        return result

    def prepare_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Prepare store specs for the store event."""
        if self._lazy_mode:
            return self._prepare_lazy_store_specs()
        else:
            return self._prepare_eager_store_specs(scheduler_output)

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[list[int], list[int], list[str]]:
        """Single-pass cursor walk: offload cached GPU blocks near eviction.

        Walks the GPU free queue from the cursor, counting blocks that are
        free-or-offloaded (safe for the allocator to evict). Stops when
        target_free blocks are covered or CPU capacity is reached.
        """
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            return [], [], []

        free_queue = gpu_pool.free_block_queue
        cpu_pool = self.cpu_block_pool
        num_cpu_free = cpu_pool.get_num_free_blocks()

        # Validate cursor: stale if block was removed from free queue.
        if self._cursor is not None and self._cursor.ref_cnt > 0:
            self._cursor = None

        # Determine start node.
        if self._cursor is None:
            node = free_queue.fake_free_list_head.next_free_block
        else:
            node = self._cursor.next_free_block

        tail = free_queue.fake_free_list_tail
        gpu_ids: list[int] = []
        block_hashes: list[bytes] = []
        covered = 0
        last_visited = self._cursor

        while (
            node is not None
            and node is not tail
            and covered < self._target_free
            and len(gpu_ids) < num_cpu_free
        ):
            last_visited = node
            bhash = node.block_hash

            if (
                bhash is not None
                and not node.is_null
                and cpu_pool.cached_block_hash_to_block.get_one_block(bhash) is None
            ):
                gpu_ids.append(node.block_id)
                block_hashes.append(bhash)

            covered += 1
            node = node.next_free_block

        self._cursor = last_visited

        # Batch-allocate CPU blocks and stamp hashes.
        if gpu_ids:
            cpu_blocks = cpu_pool.get_new_blocks(len(gpu_ids))
            cpu_ids = [blk.block_id for blk in cpu_blocks]
            for cpu_blk, bhash in zip(cpu_blocks, block_hashes):  # type: ignore[assignment]
                cpu_blk._block_hash = bhash  # type: ignore[assignment]
            # Touch GPU blocks to prevent eviction during async copy.
            gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])
        else:
            cpu_ids = []

        return gpu_ids, cpu_ids, []

    def _hoist_victims_to_head(
        self,
        n_needed: int,
        tier: str = "l1",
        block_pool: BlockPool | None = None,
    ) -> None:
        """SM70 fork (Phase E.3b/E.5b): rearrange a tier's free queue so the
        n_needed lowest-score cached blocks land at the head, where
        ``BlockPool.get_new_blocks`` will pop them first.

        Walks the free queue (cached-and-free blocks are interleaved with
        pristine-free), scores each, sorts ascending, then surgically
        re-links the lowest-N to the queue head via the linked-list ops.

        ``tier`` selects which Marconi metadata namespace to score
        against. ``block_pool`` defaults to the L1 pool to preserve
        Phase E.3b behavior; pass the L2 pool to hoist L2 victims for
        demote/drop decisions.
        """
        if self._eviction_policy is None or n_needed <= 0:
            return
        pool = block_pool if block_pool is not None else self.cpu_block_pool
        fq = pool.free_block_queue
        if fq.num_free_blocks <= n_needed:
            # Whole queue will be drained; ordering is irrelevant.
            return

        # Walk full queue collecting nodes (bounded by num_free_blocks).
        fake_tail = fq.fake_free_list_tail
        node = fq.fake_free_list_head.next_free_block
        nodes: list = []
        while node is not None and node is not fake_tail:
            nodes.append(node)
            node = node.next_free_block

        if len(nodes) <= n_needed:
            return

        policy = self._eviction_policy
        # Snapshot the queue's current head order (LRU's eviction set)
        # BEFORE sorting so we can detect "no reorder needed".
        existing_head_ids = {n.block_id for n in nodes[:n_needed]}
        # Sort ascending by score, tie-break by block_id for determinism.
        nodes.sort(key=lambda n: (policy.score(n.block_id, tier), n.block_id))
        victims = nodes[:n_needed]

        # If LRU's natural victim set equals Marconi's, no re-link needed.
        if existing_head_ids == {v.block_id for v in victims}:
            return

        # Remove victims from current positions and prepend at head.
        for v in victims:
            fq.remove(v)
            policy.forget(v.block_id, tier)

        fake_head = fq.fake_free_list_head
        after_head = fake_head.next_free_block
        prev = fake_head
        for v in victims:
            v.prev_free_block = prev
            prev.next_free_block = v
            prev = v
        prev.next_free_block = after_head
        if after_head is not None:
            after_head.prev_free_block = prev

        fq.num_free_blocks += len(victims)
        policy.hoisted += len(victims)
        policy.evictions += len(victims)

    def _demote_l1_victims_for_admission(
        self, n_needed: int
    ) -> list[tuple[int, int]]:
        """SM70 fork (Phase E.5b): before admitting n_needed new blocks
        to L1, identify which currently-cached L1 blocks would be evicted
        (they sit at the head of L1's free queue after the hoist) and
        demote them to L2 instead of dropping their hashes.

        Side-effects on the caller's behalf:
          - Allocates L2 slots (may itself evict L2 victims by score).
          - Moves the hash entry from L1's cache map to L2's cache map.
          - Resets the L1 block's hash so subsequent get_new_blocks()
            sees it as a clean slot.
          - Transfers Marconi metadata from (l1, vid) to (l2, dst).
          - Calls free_blocks on the freshly-stamped L2 block so it
            re-enters L2's free queue as cached-and-free, eligible for
            future hits.

        Returns the list of (l1_block_id, l2_block_id) demote pairs so
        the caller can emit them to the worker for memcpy.
        """
        if (
            not self._dual_tier
            or self.l2_block_pool is None
            or n_needed <= 0
        ):
            return []

        # Walk L1's free queue head, pick cached blocks that would be the
        # next get_new_blocks() victims. _hoist_victims_to_head has
        # already reordered the queue so head==Marconi worst.
        fq = self.cpu_block_pool.free_block_queue
        fake_tail = fq.fake_free_list_tail
        node = fq.fake_free_list_head.next_free_block
        victims: list = []  # KVCacheBlock objects
        walked = 0
        while (
            node is not None
            and node is not fake_tail
            and walked < n_needed
            and len(victims) < n_needed
        ):
            if node.block_hash is not None:
                victims.append(node)
            walked += 1
            node = node.next_free_block

        if not victims:
            return []

        # Make room in L2 via its own Marconi-hoisted eviction. If L2 is
        # tight, hoist worst L2 blocks to head; get_new_blocks drops them.
        self._hoist_victims_to_head(
            len(victims),
            tier="l2",
            block_pool=self.l2_block_pool,
        )
        # If L2 free queue can't satisfy len(victims), demote as many as
        # fit. The overflow falls back to the normal L1 drop path.
        l2_free = self.l2_block_pool.get_num_free_blocks()
        if l2_free < len(victims):
            victims = victims[:l2_free]
            if not victims:
                return []

        l2_blocks = self.l2_block_pool.get_new_blocks(len(victims))
        pairs: list[tuple[int, int]] = []
        policy = self._eviction_policy
        for l1_blk, l2_blk in zip(victims, l2_blocks):
            bhash = l1_blk.block_hash
            if bhash is None:
                # Race / unexpected: skip cleanly.
                self.l2_block_pool.free_blocks([l2_blk])
                continue
            # Stamp hash onto L2, register in L2 cache map.
            l2_blk._block_hash = bhash  # type: ignore[assignment]
            self.l2_block_pool.cached_block_hash_to_block.insert(
                bhash, l2_blk
            )
            # Remove hash from L1 cache map and clear L1 block's hash so
            # the next get_new_blocks() sees it as a clean slot (the
            # subsequent _maybe_evict_cached_block call is a no-op).
            self.cpu_block_pool.cached_block_hash_to_block.pop(
                bhash, l1_blk.block_id
            )
            l1_blk.reset_hash()
            # Transfer Marconi metadata from L1 namespace to L2.
            if policy is not None:
                policy.transfer_tier(
                    "l1", l1_blk.block_id, "l2", l2_blk.block_id
                )
                policy.demotes += 1
            pairs.append((l1_blk.block_id, l2_blk.block_id))
            # Drop L2 ref_cnt back to 0 so the block re-enters its free
            # queue at the tail, ready to be returned by find_longest
            # for future L2 hits.
            self.l2_block_pool.free_blocks([l2_blk])
        return pairs

    def _promote_l2_block_to_l1(
        self, l2_block: "KVCacheBlock"
    ) -> "KVCacheBlock | None":
        """SM70 fork (Phase E.5b-2): move a single L2-resident cached block
        to L1, allocating an L1 slot (with possible Marconi-driven demote
        cascade) and emitting promote/demote pairs onto the pending queues.

        Returns the new L1-resident KVCacheBlock with ref_cnt=1 (caller
        is responsible for touch/free accounting), or None if no L1 slot
        could be allocated.

        If the same hash already lives in L1's cache map (rare, can happen
        if a previous step admitted the same hash to L1 while it was also
        in L2), returns the existing L1 block and just removes the L2
        entry to restore the "hash exists in exactly one tier" invariant.
        """
        if (
            not self._dual_tier
            or self.l2_block_pool is None
            or l2_block.is_null
        ):
            return None
        bhash = l2_block.block_hash
        if bhash is None:
            return None
        # Edge case: hash already in L1. Re-use the L1 block, drop the L2
        # duplicate, and skip the memcpy.
        existing_l1 = self.cpu_block_pool.cached_block_hash_to_block.get_one_block(
            bhash
        )
        if existing_l1 is not None:
            self.l2_block_pool.cached_block_hash_to_block.pop(
                bhash, l2_block.block_id
            )
            l2_block.reset_hash()
            if self._eviction_policy is not None:
                self._eviction_policy.forget(l2_block.block_id, tier="l2")
            return existing_l1

        # Make room in L1: hoist and possibly demote the L1 victim to L2.
        self._hoist_victims_to_head(1)
        demoted = self._demote_l1_victims_for_admission(1)
        if demoted:
            self._pending_demote_pairs.extend(demoted)

        # Allocate one L1 slot. May still drop an L1 victim's hash if L2
        # was full and the demote helper couldn't relocate it.
        if self.cpu_block_pool.get_num_free_blocks() <= 0:
            return None
        l1_blocks = self.cpu_block_pool.get_new_blocks(1)
        l1_block = l1_blocks[0]
        # Stamp hash on L1; move cache-map entry from L2 to L1.
        l1_block._block_hash = bhash  # type: ignore[assignment]
        self.cpu_block_pool.cached_block_hash_to_block.insert(bhash, l1_block)
        self.l2_block_pool.cached_block_hash_to_block.pop(
            bhash, l2_block.block_id
        )
        l2_block.reset_hash()
        # Transfer Marconi metadata.
        if self._eviction_policy is not None:
            self._eviction_policy.transfer_tier(
                "l2", l2_block.block_id, "l1", l1_block.block_id
            )
            self._eviction_policy.promotes += 1
        # Emit the worker-side memcpy pair (L2 src -> L1 dst).
        self._pending_promote_pairs.append(
            (l2_block.block_id, l1_block.block_id)
        )
        return l1_block

    def _prepare_eager_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Identify newly computed blocks to offload from scheduler requests.

        Only considers blocks whose KV data has been **confirmed computed** by
        the GPU. This means blocks from the current step are NOT stored until the
        next step. If a request finishes in the same step as its last full block,
        that block may be missed. (TODO: flush on finish.)

        Returns:
            (gpu_block_ids, cpu_block_ids, req_ids) for the store event.
        """
        # SM70 fork (Phase E.3a): snapshot before-counters so we can log
        # tracker stats only when this call actually did something.
        _admit_before = self._admission_tracker.admitted
        _reject_before = self._admission_tracker.rejected

        # SM70 fork (Phase E.3b): tick step counter for age-based decay.
        if self._eviction_policy is not None:
            self._eviction_policy.tick()

        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        req_ids: list[str] = []

        gpu_block_pool = self._gpu_block_pool
        if gpu_block_pool is None:
            return [], [], []
        cpu_block_pool = self.cpu_block_pool
        num_free = cpu_block_pool.get_num_free_blocks()
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        num_groups = len(kv_cache_groups)
        gpu_blocks_this_step: set[int] = set()

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            # Accumulate new block IDs.
            if preempted:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups
            if new_block_id_groups:
                for g in range(min(num_groups, len(new_block_id_groups))):
                    if new_block_id_groups[g] is not None:
                        state.block_ids[g].extend(new_block_id_groups[g])

            num_new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_new_tokens == 0:
                continue

            block_ids_by_group = state.block_ids
            if not block_ids_by_group:
                continue

            # --- Phase 1: Scan blocks, classify as cached vs to-store ---
            gpu_block_ids: list[int] = []
            block_hashes_to_store: list[bytes] = []
            # SM70 fork (Phase E.3b): per-admitted-block depth (block index
            # within the request) and raw bhash (admission tracker key)
            # for the eviction policy.
            depths_to_record: list[int] = []
            raw_bhashes_to_record: list[bytes] = []
            advanced_per_group: list[int] = [0] * num_groups
            out_of_space = False
            # Confirmed tokens: KV data written and visible to all streams.
            req = state.request
            confirmed_tokens = req.num_computed_tokens - req.num_output_placeholders

            for g in range(num_groups):
                # FIXME (yifan): handle CPU cache eviction, where
                # num_stored_blocks can be stale and omit evicted blocks in
                # the middle of the request.
                already_stored_g = state.num_stored_blocks[g]
                group_gpu_ids = block_ids_by_group[g]

                # Cap to blocks with confirmed KV data.
                g_block_size = kv_cache_groups[g].kv_cache_spec.block_size
                ready_blocks_g = confirmed_tokens // g_block_size
                scannable = group_gpu_ids[already_stored_g:ready_blocks_g]

                for i, gpu_block_id in enumerate(scannable):
                    gpu_block = gpu_block_pool.blocks[gpu_block_id]
                    if gpu_block.is_null:
                        advanced_per_group[g] += 1
                        continue

                    bhash_with_group = gpu_block.block_hash
                    if bhash_with_group is None:
                        break

                    # Check if this group's data is already scheduled for store
                    # in this step or already cached in CPU (L1 or, in
                    # dual-tier mode, L2). The L2 check maintains the
                    # exclusivity invariant: a given hash lives in at
                    # most one tier at a time.
                    already_l2 = (
                        self._dual_tier
                        and self.l2_block_pool is not None
                        and self.l2_block_pool.cached_block_hash_to_block.get_one_block(
                            bhash_with_group
                        )
                        is not None
                    )
                    if (
                        gpu_block_id in gpu_blocks_this_step
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            bhash_with_group
                        )
                        is not None
                        or already_l2
                    ):
                        advanced_per_group[g] += 1
                        continue

                    # SM70 fork (Phase E.3a): Marconi admission gate. Skip
                    # storing prefixes the admission tracker considers
                    # unlikely to be reused. The block continues to live in
                    # GPU; if the same hash reappears in future requests,
                    # observe() bumps its count and a later iteration will
                    # admit it.
                    # bhash_with_group bundles a group_id; observe() in
                    # get_num_new_matched_tokens uses raw BlockHash. Strip
                    # so both code paths key the tracker identically.
                    raw_bhash = get_block_hash(bhash_with_group)
                    if not self._admission_tracker.should_admit(raw_bhash):
                        advanced_per_group[g] += 1
                        continue

                    if num_free <= 0:
                        out_of_space = True
                        break
                    num_free -= 1

                    gpu_block_ids.append(gpu_block_id)
                    block_hashes_to_store.append(bhash_with_group)
                    # Depth = block index within this group's portion of
                    # the request. For the FA group this maps to token
                    # offset / FA_block_size; for sharded groups it tracks
                    # the same logical position.
                    depths_to_record.append(already_stored_g + i)
                    raw_bhashes_to_record.append(raw_bhash)
                    advanced_per_group[g] += 1

                if out_of_space:
                    break

            # --- Phase 2: Batch allocate CPU blocks and stamp hashes ---
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                # SM70 fork (Phase E.3b): hoist low-score victims to the
                # head of the free queue so they're popped first by
                # get_new_blocks (replaces pure LRU with FLOP-aware order).
                self._hoist_victims_to_head(n_to_alloc)
                # SM70 fork (Phase E.5b-1): in dual-tier mode, before
                # get_new_blocks() drops the cached blocks at the L1
                # queue head, demote them to L2. The pairs are accumulated
                # for the worker to memcpy in the next step boundary.
                if self._dual_tier:
                    demoted = self._demote_l1_victims_for_admission(n_to_alloc)
                    if demoted:
                        self._pending_demote_pairs.extend(demoted)
                cpu_blocks_alloc = cpu_block_pool.get_new_blocks(n_to_alloc)
                cpu_block_ids = [blk.block_id for blk in cpu_blocks_alloc]
                for cpu_blk, bhash in zip(cpu_blocks_alloc, block_hashes_to_store):
                    cpu_blk._block_hash = bhash  # type: ignore[assignment]
                # Record (depth, raw_bhash) for the reuse-aware eviction
                # policy. raw_bhash lets score() look up admission count.
                if self._eviction_policy is not None:
                    for cpu_blk, depth, raw_bh in zip(
                        cpu_blocks_alloc,
                        depths_to_record,
                        raw_bhashes_to_record,
                    ):
                        self._eviction_policy.record_store(
                            cpu_blk.block_id, depth, raw_bh
                        )
            else:
                cpu_block_ids = []

            if cpu_block_ids:
                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)
                gpu_blocks_this_step.update(gpu_block_ids)

                # Touch GPU blocks to prevent freeing during async copy
                gpu_block_pool.touch(
                    [gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU (%d groups)",
                    req_id,
                    len(cpu_block_ids),
                    num_groups,
                )

            # Advance per-group cursors (includes cached hits + newly stored)
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]

        # SM70 fork (Phase E.3a): log tracker stats when Marconi is
        # enabled and this call admitted or rejected at least one block.
        # Default (threshold=1) path stays quiet.
        if self._admission_tracker.threshold > 1:
            admit_delta = self._admission_tracker.admitted - _admit_before
            reject_delta = self._admission_tracker.rejected - _reject_before
            if admit_delta > 0 or reject_delta > 0:
                logger.info(
                    "Marconi tracker: +admit=%d +reject=%d cumulative=%s",
                    admit_delta, reject_delta,
                    self._admission_tracker.stats(),
                )

        # SM70 fork (Phase E.3b): log eviction stats when this call hoisted
        # at least one victim. Quiet otherwise to avoid noise.
        if self._eviction_policy is not None and merged_cpu_block_ids:
            logger.info(
                "Marconi eviction: stats=%s",
                self._eviction_policy.stats(),
            )

        return merged_gpu_block_ids, merged_cpu_block_ids, req_ids

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Handle async transfer completions from worker.

        Load completions arrive via finished_recving (real req_ids).
        Store completions arrive via kv_connector_worker_meta as
        per-event worker counts. We accumulate across steps and process
        a store event only when all workers have reported completion.
        """
        # --- Load completions ---
        for req_id in list(connector_output.finished_recving or []):
            self._cleanup_load_request(req_id)

        # --- Store completions ---
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        for event_idx, count in meta.completed_store_events.items():
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total

    def _process_store_event(self, event_idx: int) -> None:
        """Process a fully-completed store event."""
        transfer = self._store_event_to_blocks.pop(event_idx)
        self._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids)
        logger.debug(
            "Store event %d completed: cached %d blocks to CPU",
            event_idx,
            len(transfer.cpu_block_ids),
        )

        # Eager only: update per-req state
        if not self._lazy_mode:
            for req_id in self._store_event_to_reqs.pop(event_idx, []):
                state = self._reqs_to_store.get(req_id)
                if state is None:
                    continue
                state.store_events.discard(event_idx)
                if state.finished and not state.store_events:
                    self._cleanup_store_request(req_id)

    def _process_store_completion(
        self, gpu_block_ids: list[int], cpu_block_ids: list[int]
    ) -> None:
        """Cache CPU blocks per-group and release GPU refs.

        Block hashes were stamped on CPU blocks at allocation time (in
        ``_prepare_*_store_specs``).  Here we just register them in the
        cache map so they become discoverable by the load path.
        """
        assert len(cpu_block_ids) == len(gpu_block_ids)

        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        for cpu_block in cpu_blocks:
            bhash = cpu_block.block_hash
            assert bhash is not None
            self.cpu_block_pool.cached_block_hash_to_block.insert(bhash, cpu_block)

        # Free CPU and GPU blocks' ref counts to turn them into prefix cache
        self.cpu_block_pool.free_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
        )

    def has_pending_stores(self) -> bool:
        """Return True if there are in-flight store transfers."""
        return bool(self._store_event_to_blocks)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id

        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
        if not self._lazy_mode:
            store_state = self._reqs_to_store.get(req_id)
            if store_state is not None:
                if store_state.store_events:
                    store_state.finished = True  # Defer: stores in-flight
                else:
                    self._cleanup_store_request(req_id)

        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids=[])

    def _cleanup_load_request(self, req_id: str) -> None:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up event mappings,
        and frees CPU/GPU touch refs.
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            return
        # Remove from load event mapping (only this req, not whole event)
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)

        if state.transfer_meta is not None:
            # Free CPU touch refs
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.cpu_block_ids
            )
            # Free GPU touch refs
            assert self._gpu_block_pool is not None
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.gpu_block_ids
            )

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        Metadata-only cleanup but no block freeing. Job completion handles
        block caching and GPU ref freeing via _process_store_completion().
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for event_idx in list(state.store_events):
            if (reqs := self._store_event_to_reqs.get(event_idx)) is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_event_to_reqs.pop(event_idx, None)
        state.store_events.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        return self.cpu_block_pool.take_events()
