# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for SimpleCPUOffloadConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

INVALID_JOB_ID = -1


@dataclass
class SimpleCPUOffloadMetadata(KVConnectorMetadata):
    """
    Metadata passed from scheduler to worker for CPU offload operations.

    The worker receives flat block lists keyed by a monotonic event_idx.
    Job->req_id translation is handled by the scheduler-side manager
    (via inverse maps), so the worker never knows about request identities.
    """

    # Load event per step. INVALID_JOB_ID means no blocks to load this step.
    # The existing fields target the L1 tier (or single-tier in legacy mode).
    load_event: int = INVALID_JOB_ID
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_cpu_blocks: list[int] = field(default_factory=list)
    # SM70 fork (Phase E.5a): L2 tier transfer pairs. Empty in single-tier
    # mode. Share the same load_event with L1 entries when both are issued
    # by the same scheduler step.
    load_gpu_blocks_l2: list[int] = field(default_factory=list)
    load_cpu_blocks_l2: list[int] = field(default_factory=list)
    # Reverse map: load_event->req_ids, for tracking requests with finished load events
    load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)

    # Store event per step. INVALID_JOB_ID means no blocks to store this step.
    # The existing fields target the L1 tier (or single-tier in legacy mode).
    store_event: int = INVALID_JOB_ID
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_cpu_blocks: list[int] = field(default_factory=list)
    # SM70 fork (Phase E.5a): direct-to-L2 stores. Usually empty because
    # admissions land in L1 first; populated only if the manager bypasses
    # L1 (e.g., very large, very cold admit).
    store_gpu_blocks_l2: list[int] = field(default_factory=list)
    store_cpu_blocks_l2: list[int] = field(default_factory=list)

    # SM70 fork (Phase E.5a): CPU-side inter-tier moves issued this step.
    #   demote_pairs:  list of (l1_block_id, l2_block_id) — copy L1 -> L2
    #   promote_pairs: list of (l2_block_id, l1_block_id) — copy L2 -> L1
    # The worker performs these as synchronous CPU memcpy.
    demote_pairs: list = field(default_factory=list)
    promote_pairs: list = field(default_factory=list)

    # Whether any requests were preempted this step and need flush pending transfers.
    need_flush: bool = False


@dataclass
class SimpleCPUOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store events.

    Each worker reports {event_idx: 1} for newly completed stores.
    ``aggregate()`` sums counts across workers within a step.
    The scheduler-side manager accumulates across steps and processes
    a store completion only when count reaches ``world_size``.
    """

    completed_store_events: dict[int, int]

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, SimpleCPUOffloadWorkerMetadata)
        merged = dict(self.completed_store_events)
        for k, v in other.completed_store_events.items():
            merged[k] = merged.get(k, 0) + v
        return SimpleCPUOffloadWorkerMetadata(completed_store_events=merged)
