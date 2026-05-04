# W8A16 weight-only INT8 quantization for SM70 (V100).
# Quantizes FP16 weights to INT8 + FP16 scales at load time.
# Supports both online quantization (FP16 checkpoint) and
# offline quantized checkpoints (INT8-as-FP16 + .scale tensors).

from typing import Optional

import torch
import torch.nn as nn

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)

GROUP_SIZE = 128


class W8A16SM70Config(QuantizationConfig):
    """Config class for W8A16 weight-only INT8 quantization on SM70."""

    def __init__(self, group_size: int = GROUP_SIZE, pre_quantized: bool = False):
        self.group_size = group_size
        self.pre_quantized = pre_quantized

    @classmethod
    def get_name(cls) -> str:
        return "w8a16_sm70"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config) -> "W8A16SM70Config":
        group_size = config.get("group_size", GROUP_SIZE)
        pre_quantized = config.get("pre_quantized", False)
        return cls(group_size=group_size, pre_quantized=pre_quantized)

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> Optional["W8A16SM70LinearMethod"]:
        if isinstance(layer, LinearBase):
            return W8A16SM70LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


def quantize_weights_grouped(
    weight: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP16 weight [N, K] to INT8 + per-group FP16 scales."""
    n, k = weight.shape
    num_groups = k // group_size
    weight_2d = weight.reshape(n, num_groups, group_size)
    max_abs = weight_2d.abs().amax(dim=2, keepdim=True).clamp(min=1e-5)
    scale = (max_abs / 127.0).squeeze(2).to(torch.float16)  # [N, num_groups]
    weight_q = (weight_2d / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
    weight_int8 = weight_q.reshape(n, k)
    return weight_int8, scale.t().contiguous()  # [num_groups, N]


class W8A16SM70LinearMethod(LinearMethodBase):
    """Linear method for W8A16 SM70 quantization.

    Supports two modes:
    - Online quantization: FP16 checkpoint, quantize at load time
    - Offline quantized: INT8-as-FP16 weights + .scale tensors in checkpoint
    """

    def __init__(self, quant_config: W8A16SM70Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quant_config.pre_quantized:
            num_groups = input_size_per_partition // self.quant_config.group_size
            scales = GroupQuantScaleParameter(
                data=torch.empty(
                    num_groups,
                    output_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=0,
                output_dim=1,
                weight_loader=weight_loader,
            )
            layer.register_parameter("scale", scales)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        device = layer.weight.data.device
        cap = torch.cuda.get_device_capability(device)
        is_sm70 = cap[0] == 7 and cap[1] == 0
        has_prepare = hasattr(torch.ops._C, "w8a16_sm70_prepare")

        if self.quant_config.pre_quantized:
            self._process_pre_quantized(layer, device, is_sm70, has_prepare)
        else:
            self._process_online(layer, device, is_sm70, has_prepare)

    def _process_pre_quantized(
        self,
        layer: nn.Module,
        device: torch.device,
        is_sm70: bool,
        has_prepare: bool,
    ) -> None:
        weight_data = layer.weight.data  # [N, K]
        n, k = weight_data.shape
        group_size = self.quant_config.group_size

        # Layers skipped by offline quantization (TP-incompatible dims)
        # retain original FP16 weights and have no valid scale.
        can_quantize = k % group_size == 0 and n % 8 == 0
        if not can_quantize:
            # Weight is already FP16 — nothing to do
            return

        scale_data = layer.scale.data  # [num_groups, N]

        if is_sm70 and has_prepare:
            weight_int8 = weight_data.round().clamp(-128, 127).to(torch.int8)

            n_padded = ((n + 63) // 64) * 64
            if n_padded != n:
                w = torch.zeros(n_padded, k, dtype=torch.int8, device=device)
                w[:n] = weight_int8
                s = torch.zeros(
                    scale_data.shape[0], n_padded,
                    dtype=torch.float16, device=device,
                )
                s[:, :n] = scale_data
            else:
                w = weight_int8
                s = scale_data

            tm_weight, tm_scales, meta = ops.w8a16_sm70_prepare(
                w, s, self.quant_config.group_size
            )
            layer._w8a16_sm70_weight = tm_weight
            layer._w8a16_sm70_scales = tm_scales
            layer._w8a16_sm70_w_ld = int(meta[0])
            layer._w8a16_sm70_s_ld = int(meta[1])
            layer._w8a16_sm70_n_orig = n
            layer._w8a16_sm70_prepared = True

            layer.weight = nn.Parameter(
                torch.empty(0, dtype=torch.float16, device=device),
                requires_grad=False,
            )
        else:
            # Non-SM70 or incompatible shape: dequantize back to FP16
            num_groups = scale_data.shape[0]
            group_size = k // num_groups
            dequant = (
                weight_data.float()
                .reshape(n, num_groups, group_size)
                .mul_(scale_data.t().unsqueeze(2).float())
                .reshape(n, k)
                .to(torch.float16)
            )
            layer.weight = nn.Parameter(dequant, requires_grad=False)

        layer.scale = nn.Parameter(
            torch.empty(0, dtype=torch.float16, device=device),
            requires_grad=False,
        )

    def _process_online(
        self,
        layer: nn.Module,
        device: torch.device,
        is_sm70: bool,
        has_prepare: bool,
    ) -> None:
        if not is_sm70 or not has_prepare:
            return

        weight_data = layer.weight.data
        n, k = weight_data.shape
        if k % self.quant_config.group_size != 0:
            return
        if n % 8 != 0:
            return

        n_padded = ((n + 63) // 64) * 64
        if n_padded != n:
            w = torch.zeros(n_padded, k, dtype=torch.float16, device=device)
            w[:n] = weight_data
        else:
            w = weight_data

        weight_int8, scales_fp16 = quantize_weights_grouped(
            w, self.quant_config.group_size
        )

        tm_weight, tm_scales, meta = ops.w8a16_sm70_prepare(
            weight_int8, scales_fp16, self.quant_config.group_size
        )

        layer._w8a16_sm70_weight = tm_weight
        layer._w8a16_sm70_scales = tm_scales
        layer._w8a16_sm70_w_ld = int(meta[0])
        layer._w8a16_sm70_s_ld = int(meta[1])
        layer._w8a16_sm70_n_orig = n
        layer._w8a16_sm70_prepared = True

        layer.weight = nn.Parameter(
            torch.empty(0, dtype=torch.float16, device=device),
            requires_grad=False,
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])

        if getattr(layer, "_w8a16_sm70_prepared", False):
            n_padded = layer._w8a16_sm70_weight.size(0)
            n_orig = layer._w8a16_sm70_n_orig
            out = torch.empty(
                (reshaped_x.size(0), n_padded),
                dtype=torch.float16,
                device=reshaped_x.device,
            )
            ops.w8a16_sm70_gemm_out(
                out,
                reshaped_x,
                layer._w8a16_sm70_weight,
                layer._w8a16_sm70_scales,
                self.quant_config.group_size,
                layer._w8a16_sm70_w_ld,
                layer._w8a16_sm70_s_ld,
                False,
            )
            return out[:, :n_orig].reshape(x.shape[:-1] + (n_orig,))

        # Fallback: FP16 matmul
        out = torch.mm(reshaped_x, layer.weight.t())
        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (out.shape[-1],))
