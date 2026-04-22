# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE


@dataclass(frozen=True)
class SM70DecodeFastPathDecision:
    eligible: bool
    reason: str


@dataclass(frozen=True)
class SM70CompactRouting:
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor


def describe_sm70_decode_fastpath(
    layer: "FusedMoE",
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
) -> SM70DecodeFastPathDecision:
    if not current_platform.is_cuda():
        return SM70DecodeFastPathDecision(False, "platform_not_cuda")
    if not current_platform.is_device_capability((7, 0)):
        return SM70DecodeFastPathDecision(False, "device_not_sm70")
    if x.device.type != "cuda":
        return SM70DecodeFastPathDecision(False, "hidden_states_not_cuda")
    if x.dtype != torch.float16:
        return SM70DecodeFastPathDecision(False, "hidden_states_not_fp16")
    if x.ndim != 2:
        return SM70DecodeFastPathDecision(False, "hidden_states_not_2d")
    if x.shape[0] > 8:
        return SM70DecodeFastPathDecision(False, "token_count_above_small_m_limit")
    if topk_weights.device != x.device or topk_ids.device != x.device:
        return SM70DecodeFastPathDecision(False, "routing_tensors_not_on_same_device")
    if shared_experts_input is not None:
        return SM70DecodeFastPathDecision(False, "shared_experts_input_present")
    if layer.expert_map is not None:
        return SM70DecodeFastPathDecision(False, "expert_map_present")
    return SM70DecodeFastPathDecision(True, "eligible_noop_stub")


def should_use_sm70_compact_routing(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> SM70DecodeFastPathDecision:
    if os.getenv("VLLM_SM70_ENABLE_COMPACT_ROUTING") != "1":
        return SM70DecodeFastPathDecision(False, "compact_routing_disabled")
    if not current_platform.is_cuda():
        return SM70DecodeFastPathDecision(False, "platform_not_cuda")
    if not current_platform.is_device_capability((7, 0)):
        return SM70DecodeFastPathDecision(False, "device_not_sm70")
    if hidden_states.device.type != "cuda":
        return SM70DecodeFastPathDecision(False, "hidden_states_not_cuda")
    if hidden_states.dtype != torch.float16:
        return SM70DecodeFastPathDecision(False, "hidden_states_not_fp16")
    if hidden_states.ndim != 2:
        return SM70DecodeFastPathDecision(False, "hidden_states_not_2d")
    if hidden_states.shape[0] > 8:
        return SM70DecodeFastPathDecision(False, "token_count_above_small_m_limit")
    if topk_ids.device != hidden_states.device:
        return SM70DecodeFastPathDecision(False, "topk_ids_not_on_same_device")
    if topk_ids.ndim != 2:
        return SM70DecodeFastPathDecision(False, "topk_ids_not_2d")
    if topk_ids.numel() == 0:
        return SM70DecodeFastPathDecision(False, "empty_topk_ids")
    if expert_map is not None:
        return SM70DecodeFastPathDecision(False, "expert_map_present")
    return SM70DecodeFastPathDecision(True, "eligible_compact_routing")


def build_sm70_compact_routing(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> SM70CompactRouting:
    if topk_ids.ndim != 2:
        raise ValueError(f"Expected 2D topk_ids, got shape={tuple(topk_ids.shape)}")

    flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
    num_slots = flat_topk_ids.numel()
    device = topk_ids.device

    if num_slots == 0:
        raise ValueError("topk_ids must contain at least one routed expert")

    slot_ids = torch.arange(num_slots, device=device, dtype=torch.int32)
    counts = torch.bincount(flat_topk_ids, minlength=num_experts)
    active_experts = torch.nonzero(counts, as_tuple=False).flatten().to(torch.int32)
    active_counts = counts.index_select(0, active_experts.to(torch.int64)).to(torch.int32)

    padded_counts = ((active_counts + block_size - 1) // block_size) * block_size
    total_padded = int(padded_counts.sum().item())
    num_blocks = total_padded // block_size
    invalid_slot = num_slots

    order = torch.argsort(flat_topk_ids, stable=True)
    sorted_slots = slot_ids.index_select(0, order)

    segment_offsets = torch.cumsum(padded_counts, dim=0) - padded_counts
    segment_offsets = segment_offsets.to(torch.int32)
    source_offsets = torch.cumsum(active_counts, dim=0) - active_counts
    source_offsets = source_offsets.to(torch.int32)

    output_positions = torch.arange(total_padded, device=device, dtype=torch.int32)
    output_segments = torch.repeat_interleave(
        torch.arange(active_experts.numel(), device=device, dtype=torch.int32),
        padded_counts,
    )
    within_segment = output_positions - segment_offsets.index_select(0, output_segments)
    real_counts = active_counts.index_select(0, output_segments)
    is_real = within_segment < real_counts

    sorted_token_ids = torch.full(
        (total_padded,),
        invalid_slot,
        dtype=torch.int32,
        device=device,
    )
    source_indices = source_offsets.index_select(0, output_segments) + within_segment
    sorted_token_ids[is_real] = sorted_slots.index_select(0, source_indices[is_real])

    expert_ids = torch.repeat_interleave(active_experts, padded_counts // block_size)
    num_tokens_post_padded = torch.tensor([total_padded], dtype=torch.int32, device=device)

    if expert_ids.numel() != num_blocks:
        raise RuntimeError(
            "Compact routing constructed inconsistent block metadata: "
            f"expert_ids={expert_ids.numel()} num_blocks={num_blocks}"
        )

    return SM70CompactRouting(
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


def maybe_apply_sm70_decode_fastpath(
    layer: "FusedMoE",
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    _ = describe_sm70_decode_fastpath(
        layer=layer,
        x=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts_input=shared_experts_input,
    )
    return None
