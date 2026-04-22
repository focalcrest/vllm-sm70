# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE


@dataclass(frozen=True)
class SM70DecodeFastPathDecision:
    eligible: bool
    reason: str


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
