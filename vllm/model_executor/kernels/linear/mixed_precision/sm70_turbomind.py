# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

SM70_W4_GROUP_SIZES = (32, 64, 128)

# AWQ "store" permutation: AWQ-format storage position i contains the
# linear-order nibble at index AWQ_PERM[i] (the AutoAWQ even-then-odd
# ordering used for efficient SIMT dequant).
_AWQ_PERM = [0, 2, 4, 6, 1, 3, 5, 7]


def _ct_w4_to_classic_awq(
    weight_packed: torch.Tensor,       # [N, K//8] int32, linear-K packed
    weight_zero_point: torch.Tensor,   # [N//8, num_groups] int32, linear-N packed
    weight_scale: torch.Tensor,        # [N, num_groups] fp16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert compressed-tensors W4 tensors to classic AWQ format.

    Classic AWQ layout (what `ops.awq_sm70_prepare` consumes):
      - qweight: [K, N//8] int32, AWQ-interleaved along N
      - qzeros:  [num_groups, N//8] int32, AWQ-interleaved along N
      - scales:  [num_groups, N] fp16

    Verified bit-identical via dequant comparison in
    `scripts/test_w4_layout_conversion.py`.
    """
    n, k_div8 = weight_packed.shape
    k = k_div8 * 8
    num_groups = weight_scale.shape[1]
    device = weight_packed.device
    perm = torch.tensor(_AWQ_PERM, dtype=torch.long, device=device)

    # ---- weight: [N, K//8] linear-K → [K, N//8] AWQ-N ----
    w_nibbles = torch.empty(n, k, dtype=torch.uint8, device=device)
    for i in range(8):
        w_nibbles[:, i::8] = ((weight_packed >> (i * 4)) & 0xF).to(torch.uint8)
    w_KN = w_nibbles.t().contiguous()
    w_awq_grouped = w_KN.reshape(k, n // 8, 8).index_select(2, perm)
    qweight = torch.zeros(k, n // 8, dtype=torch.int32, device=device)
    for i in range(8):
        qweight |= w_awq_grouped[:, :, i].to(torch.int32) << (i * 4)

    # ---- zeros: [N//8, num_groups] linear-N → [num_groups, N//8] AWQ-N ----
    z_nibbles = torch.empty(n, num_groups, dtype=torch.uint8, device=device)
    for i in range(8):
        z_nibbles[i::8, :] = ((weight_zero_point >> (i * 4)) & 0xF).to(torch.uint8)
    z_GN = z_nibbles.t().contiguous()
    z_awq_grouped = z_GN.reshape(num_groups, n // 8, 8).index_select(2, perm)
    qzeros = torch.zeros(num_groups, n // 8, dtype=torch.int32, device=device)
    for i in range(8):
        qzeros |= z_awq_grouped[:, :, i].to(torch.int32) << (i * 4)

    # ---- scales: [N, num_groups] → [num_groups, N] ----
    scales = weight_scale.t().contiguous()

    return qweight, qzeros, scales


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

        if c.weight_type != scalar_types.uint4:
            return False, (
                f"SM70 TurboMind kernel only supports uint4 (asym W4), "
                f"got {c.weight_type}"
            )

        if c.group_size not in SM70_W4_GROUP_SIZES:
            return False, (
                f"SM70 W4 supports group_size in {SM70_W4_GROUP_SIZES}, "
                f"got {c.group_size}"
            )

        if not c.zero_points:
            # awq_sm70_prepare requires qzeros. Sym W4 (uint4b8) would need
            # a different code path; not implemented here.
            return False, "SM70 W4 path requires asymmetric (zero_points)"

        if c.has_g_idx:
            return False, "SM70 TurboMind kernel does not support group indices"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """compressed-tensors W4 → classic AWQ format → awq_sm70_prepare.

        After this returns the layer carries `_awq_sm70_*` tensors (matching
        the convention used by the existing AWQ scheme's SM70 fast-path in
        `awq.py:317`). `apply_weights` dispatches on those tensors.
        """
        device = getattr(layer, self.w_q_name).device
        c = self.config
        n = c.partition_weight_shape[1]  # output features
        k = c.partition_weight_shape[0]  # input features
        # awq_sm70_prepare requires K and N to be multiples of 8 (pack factor
        # for uint4 weights). Qwen3.6-27B's per-rank N is always divisible by
        # 8 at TP ∈ {1,2,4,8}, but assert explicitly so misuse is loud.
        assert n % 8 == 0 and k % 8 == 0, (
            f"SM70 W4 path needs K and N divisible by 8, got K={k} N={n}"
        )

        # Normalize the parameter layouts.
        w_packed = getattr(layer, self.w_q_name)
        assert isinstance(w_packed, BasevLLMParameter)
        # compressed-tensors W4 weight_packed is [N, K//8] int32 packed
        # along K with packed_factor=8 (8 uint4 per int32). Normalise to
        # input_dim=1 / output_dim=0 / packed_dim=1.
        permute_param_layout_(w_packed, input_dim=1, output_dim=0, packed_dim=1)
        weight_packed = w_packed.data.contiguous()  # [N, K//8] int32

        w_scale = getattr(layer, self.w_s_name)
        assert isinstance(w_scale, BasevLLMParameter)
        permute_param_layout_(w_scale, input_dim=1, output_dim=0)
        weight_scale = w_scale.data.contiguous()  # [N, num_groups] fp16

        assert c.zero_points and self.w_zp_name, (
            "W4 path requires asymmetric (zero_points)"
        )
        w_zp = getattr(layer, self.w_zp_name)
        assert isinstance(w_zp, BasevLLMParameter)
        # compressed-tensors zp is [N//8, num_groups] int32 packed along N.
        permute_param_layout_(w_zp, input_dim=1, output_dim=0, packed_dim=0)
        weight_zero_point = w_zp.data.contiguous()  # [N//8, num_groups] int32

        # Layout convert → classic AWQ. Verified bit-equivalent against
        # the compressed-tensors dequant in scripts/test_w4_layout_conversion.py.
        qweight, qzeros, scales = _ct_w4_to_classic_awq(
            weight_packed, weight_zero_point, weight_scale
        )

        # Feed the kernel-native AWQ-W4 prepare op.
        tm_weight, tm_scales, meta = ops.awq_sm70_prepare(
            qweight, scales, qzeros, c.group_size, False
        )
        layer.register_parameter(
            "_awq_sm70_weight",
            torch.nn.Parameter(tm_weight, requires_grad=False),
        )
        layer.register_parameter(
            "_awq_sm70_scales",
            torch.nn.Parameter(tm_scales, requires_grad=False),
        )
        layer._awq_sm70_k_ld = int(meta[0])
        layer._awq_sm70_q_ld = int(meta[1])
        layer._awq_sm70_n_orig = n

        # Free original packed weights / scales / zeros.
        for name in (self.w_q_name, self.w_s_name, self.w_zp_name):
            if not name:
                continue
            self._transform_param(
                layer, name,
                lambda _: torch.empty(0, dtype=torch.float16, device=device),
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Kernel-native AWQ-W4 via ops.awq_gemm_sm70 (Config_U4_g).
        reshaped_x = x.reshape(-1, x.shape[-1])
        n_orig = layer._awq_sm70_n_orig
        out = ops.awq_gemm_sm70(
            reshaped_x,
            layer._awq_sm70_weight,
            layer._awq_sm70_scales,
            self.config.group_size,
            layer._awq_sm70_k_ld,
            layer._awq_sm70_q_ld,
        )
        return out[:, :n_orig].reshape(x.shape[:-1] + (n_orig,))
