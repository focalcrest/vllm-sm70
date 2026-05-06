# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

SM70_GROUP_SIZE = 128
SM70_N_PAD = 64


class SM70TurboMindLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "SM70 TurboMind kernel only supported on CUDA"

        cap = current_platform.get_device_capability()
        if cap != (7, 0):
            return False, "SM70 TurboMind kernel only supported on SM70 (V100)"

        if c.weight_type != scalar_types.uint8b128:
            return False, (
                f"SM70 TurboMind kernel only supports uint8b128, got {c.weight_type}"
            )

        if c.group_size != SM70_GROUP_SIZE:
            return False, (
                f"SM70 TurboMind kernel only supports group_size={SM70_GROUP_SIZE}, "
                f"got {c.group_size}"
            )

        if c.zero_points:
            return False, "SM70 TurboMind kernel does not support zero points"

        if c.has_g_idx:
            return False, "SM70 TurboMind kernel does not support group indices"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config
        n = c.partition_weight_shape[1]  # output features
        k = c.partition_weight_shape[0]  # input features

        # Unpack weight_packed: int32 [N, K//4] → int8 [N, K]
        w_packed = getattr(layer, self.w_q_name)
        assert isinstance(w_packed, BasevLLMParameter)
        permute_param_layout_(w_packed, input_dim=1, output_dim=0, packed_dim=1)
        packed_data = w_packed.data.contiguous()  # [N, K//4] int32

        # Unpack 4 uint8 values from each int32
        shifts = torch.tensor([0, 8, 16, 24], device=device, dtype=torch.int32)
        unpacked = torch.zeros(n, k, dtype=torch.uint8, device=device)
        for i, shift in enumerate(shifts):
            unpacked[:, i::4] = ((packed_data >> shift) & 0xFF).to(torch.uint8)

        # uint8b128 → signed int8: subtract 128
        weight_int8 = unpacked.to(torch.int16) - 128
        weight_int8 = weight_int8.to(torch.int8)

        # Transpose weight_scale: fp16 [N, num_groups] → [num_groups, N]
        w_scale = getattr(layer, self.w_s_name)
        assert isinstance(w_scale, BasevLLMParameter)
        permute_param_layout_(w_scale, input_dim=1, output_dim=0)
        scales = w_scale.data.contiguous().t()  # [num_groups, N]

        # Pad N to multiple of SM70_N_PAD for TurboMind HMMA kernel
        n_padded = ((n + SM70_N_PAD - 1) // SM70_N_PAD) * SM70_N_PAD
        if n_padded != n:
            weight_int8 = torch.nn.functional.pad(
                weight_int8, (0, 0, 0, n_padded - n)
            )
            scales = torch.nn.functional.pad(scales, (0, n_padded - n))

        # Prepare weights for TurboMind HMMA_884 kernel
        tm_weight, tm_scales, meta = ops.w8a16_sm70_prepare(
            weight_int8, scales, c.group_size
        )

        # Store prepared tensors as layer attributes
        layer.register_parameter(
            "_sm70_tm_weight",
            torch.nn.Parameter(tm_weight, requires_grad=False),
        )
        layer.register_parameter(
            "_sm70_tm_scales",
            torch.nn.Parameter(tm_scales, requires_grad=False),
        )
        layer._sm70_tm_w_ld = int(meta[0])
        layer._sm70_tm_s_ld = int(meta[1])
        layer._sm70_tm_n_orig = n

        # Free original packed weights and scales
        self._transform_param(
            layer,
            self.w_q_name,
            lambda _: torch.empty(0, dtype=torch.float16, device=device),
        )
        self._transform_param(
            layer,
            self.w_s_name,
            lambda _: torch.empty(0, dtype=torch.float16, device=device),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_padded = layer._sm70_tm_weight.size(0)
        n_orig = layer._sm70_tm_n_orig

        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch.empty(
            (reshaped_x.size(0), n_padded),
            dtype=torch.float16,
            device=reshaped_x.device,
        )

        ops.w8a16_sm70_gemm_out(
            out,
            reshaped_x,
            layer._sm70_tm_weight,
            layer._sm70_tm_scales,
            self.config.group_size,
            layer._sm70_tm_w_ld,
            layer._sm70_tm_s_ld,
            False,  # gated_silu
        )

        return out[:, :n_orig].reshape(x.shape[:-1] + (n_orig,))
