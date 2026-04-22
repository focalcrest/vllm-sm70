# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

logger = init_logger(__name__)

SM70_DECODE_FASTPATH_ENABLED = (
    os.getenv("VLLM_SM70_ENABLE_UNQUANT_MOE_DECODE_FASTPATH", "0") == "1"
)
SM70_DECODE_FASTPATH_MAX_M = int(
    os.getenv("VLLM_SM70_UNQUANT_MOE_DECODE_FASTPATH_MAX_M", "8")
)


@dataclass(frozen=True)
class SM70DecodeFastPathDecision:
    eligible: bool
    reason: str


@dataclass(frozen=True)
class SM70PreparedDenseWeight:
    tm_weight: torch.Tensor
    k_ld: int


def _is_sm70_fp16_tensor(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.float16
        and torch.cuda.get_device_capability(x.device) == (7, 0)
    )


def _maybe_prepare_sm70_weight(weight: torch.Tensor) -> SM70PreparedDenseWeight | None:
    if not _is_sm70_fp16_tensor(weight):
        return None
    if weight.ndim != 2:
        return None
    if weight.stride(-1) != 1:
        weight = weight.contiguous()
    if not hasattr(torch.ops._C, "sm70_f16_prepare"):
        return None

    prepared = ops.sm70_f16_prepare(weight)
    return SM70PreparedDenseWeight(
        tm_weight=prepared[0],
        k_ld=int(prepared[1][0].item()),
    )


def maybe_prepare_sm70_decode_fastpath(layer: "FusedMoE") -> None:
    """Prepare per-expert SM70 fp16 metadata for a future decode fast path.

    This is intentionally preparation-only for now. Execution still falls back
    to the existing Triton unquantized MoE path until the decode fast path is
    implemented.
    """
    if not current_platform.is_cuda_alike():
        return
    if not hasattr(torch.ops._C, "sm70_f16_prepare"):
        return
    if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
        return
    if getattr(layer, "_sm70_decode_fastpath_prepared", False):
        return

    w13 = layer.w13_weight
    w2 = layer.w2_weight
    if not _is_sm70_fp16_tensor(w13) or not _is_sm70_fp16_tensor(w2):
        return
    if w13.ndim != 3 or w2.ndim != 3:
        return
    if layer.expert_map is not None:
        return
    if not layer.moe_config.is_act_and_mul:
        return

    num_experts, w13_n, w13_k = w13.shape
    _, w2_n, w2_k = w2.shape

    if w13_k != layer.hidden_dim or w2_n != layer.hidden_dim:
        return
    if (w13_k % 16) != 0 or (w13_n % 32) != 0:
        return
    if (w2_k % 16) != 0 or (w2_n % 32) != 0:
        return

    prepared_w13: list[SM70PreparedDenseWeight] = []
    prepared_w2: list[SM70PreparedDenseWeight] = []
    for expert_idx in range(num_experts):
        w13_prepared = _maybe_prepare_sm70_weight(w13[expert_idx])
        w2_prepared = _maybe_prepare_sm70_weight(w2[expert_idx])
        if w13_prepared is None or w2_prepared is None:
            return
        prepared_w13.append(w13_prepared)
        prepared_w2.append(w2_prepared)

    layer._sm70_decode_fastpath_num_experts = num_experts
    layer._sm70_decode_fastpath_top_k = layer.top_k
    layer._sm70_decode_fastpath_hidden = layer.hidden_dim
    layer._sm70_decode_fastpath_intermediate = w2_k
    layer._sm70_decode_fastpath_w13 = prepared_w13
    layer._sm70_decode_fastpath_w2 = prepared_w2
    layer._sm70_decode_fastpath_gate_up = torch.empty(
        (layer.top_k, w13_n), dtype=torch.float16, device=w13.device
    )
    layer._sm70_decode_fastpath_activated = torch.empty(
        (layer.top_k, w2_k), dtype=torch.float16, device=w13.device
    )
    layer._sm70_decode_fastpath_down = torch.empty(
        (layer.top_k, layer.hidden_dim), dtype=torch.float16, device=w13.device
    )
    layer._sm70_decode_fastpath_weighted = torch.empty(
        (1, layer.hidden_dim), dtype=torch.float16, device=w13.device
    )
    layer._sm70_decode_fastpath_out = torch.empty(
        (1, layer.hidden_dim), dtype=torch.float16, device=w13.device
    )
    layer._sm70_decode_fastpath_topk_weights_fp16 = torch.empty(
        (1, layer.top_k), dtype=torch.float16, device=w13.device
    )
    layer._sm70_decode_fastpath_prepared = True
    logger.info_once("SM70 unquantized MoE decode fastpath weights prepared.")


def describe_sm70_decode_fastpath(
    layer: "FusedMoE",
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
) -> SM70DecodeFastPathDecision:
    if not SM70_DECODE_FASTPATH_ENABLED:
        return SM70DecodeFastPathDecision(False, "feature_disabled")
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
    if x.shape[0] > SM70_DECODE_FASTPATH_MAX_M:
        return SM70DecodeFastPathDecision(False, "token_count_above_small_m_limit")
    if topk_weights.device != x.device or topk_ids.device != x.device:
        return SM70DecodeFastPathDecision(False, "routing_tensors_not_on_same_device")
    if shared_experts_input is not None:
        return SM70DecodeFastPathDecision(False, "shared_experts_input_present")
    if layer.expert_map is not None:
        return SM70DecodeFastPathDecision(False, "expert_map_present")
    if not getattr(layer, "_sm70_decode_fastpath_prepared", False):
        return SM70DecodeFastPathDecision(False, "weights_not_prepared")
    if x.shape[0] != 1:
        return SM70DecodeFastPathDecision(False, "m1_only_initial_version")
    return SM70DecodeFastPathDecision(True, "eligible_m1_stub")


def maybe_apply_sm70_decode_fastpath(
    layer: "FusedMoE",
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    decision = describe_sm70_decode_fastpath(
        layer=layer,
        x=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts_input=shared_experts_input,
    )
    if not decision.eligible:
        return None

    top_k = layer._sm70_decode_fastpath_top_k
    gate_up = layer._sm70_decode_fastpath_gate_up
    activated = layer._sm70_decode_fastpath_activated
    down = layer._sm70_decode_fastpath_down
    weighted = layer._sm70_decode_fastpath_weighted
    out = layer._sm70_decode_fastpath_out
    weights_fp16 = layer._sm70_decode_fastpath_topk_weights_fp16

    out.zero_()
    weights_fp16.copy_(topk_weights)

    for route_idx in range(top_k):
        expert_id = int(topk_ids[0, route_idx].item())
        w13 = layer._sm70_decode_fastpath_w13[expert_id]
        w2 = layer._sm70_decode_fastpath_w2[expert_id]

        gate_up_row = gate_up[route_idx : route_idx + 1]
        activated_row = activated[route_idx : route_idx + 1]
        down_row = down[route_idx : route_idx + 1]

        ops.sm70_f16_gemm_out(gate_up_row, x, w13.tm_weight, w13.k_ld, False)
        torch.ops._C.silu_and_mul(activated_row, gate_up_row)
        ops.sm70_f16_gemm_out(down_row, activated_row, w2.tm_weight, w2.k_ld, False)
        torch.mul(down_row, weights_fp16[:, route_idx : route_idx + 1], out=weighted)
        out.add_(weighted)

    return out
