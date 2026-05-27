# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

SM70_W4_GROUP_SIZES = (32, 64, 128)
SM70_W8_GROUP_SIZES = (128,)  # only group_size=128 is registered in sm70_884_u8a.cu

# AWQ "store" permutation: AWQ-format storage position i contains the
# linear-order nibble at index AWQ_PERM[i] (the AutoAWQ even-then-odd
# ordering used for efficient SIMT dequant).
_AWQ_PERM = [0, 2, 4, 6, 1, 3, 5, 7]


def _ct_w8_unpack(
    weight_packed: torch.Tensor,       # [N, K//4] int32, linear-K packed (4 uint8 / int32)
    weight_zero_point: torch.Tensor,   # [N//4, num_groups] int32, linear-N packed
    weight_scale: torch.Tensor,        # [N, num_groups] fp16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack compressed-tensors W8 asym tensors into the layout the
    SM70 W8A16 kernel-native prepare expects.

    Returns (weight_int8, scales, zeros_fp16_signed) where:
      - weight_int8: [N, K] int8 with values = uint8 - 128 (signed).
      - scales:      [num_groups, N] fp16 (transposed from compressed-tensors).
      - zeros_fp16_signed: [num_groups, N] fp16 with values
                           = (zp_uint8 - 128) cast to fp16. The kernel-side
                           bias is computed via -zeros * scales, so passing
                           the signed zp makes `(q - zp) * scale` numerically
                           exact after the int8→fp16 cast.

    See `_ct_w4_to_classic_awq` for the W4 sibling.
    """
    n, k_div4 = weight_packed.shape
    k = k_div4 * 4
    num_groups = weight_scale.shape[1]
    device = weight_packed.device

    # ---- weight: [N, K//4] linear-K → [N, K] uint8 → int8 (signed) ----
    w_uint8 = torch.empty(n, k, dtype=torch.uint8, device=device)
    for i in range(4):
        w_uint8[:, i::4] = ((weight_packed >> (i * 8)) & 0xFF).to(torch.uint8)
    # Shift -128 via cast through int16. uint8 256-wide → int8 [-128, 127].
    weight_int8 = (w_uint8.to(torch.int16) - 128).to(torch.int8).contiguous()

    # ---- zp: [N//4, num_groups] linear-N → [num_groups, N] fp16 signed ----
    zp_uint8 = torch.empty(n, num_groups, dtype=torch.uint8, device=device)
    for i in range(4):
        zp_uint8[i::4, :] = ((weight_zero_point >> (i * 8)) & 0xFF).to(torch.uint8)
    # Signed zp_int8 → fp16, transposed to [num_groups, N].
    zeros_fp16 = (zp_uint8.to(torch.int16) - 128).t().contiguous().to(torch.float16)

    # ---- scales: [N, num_groups] → [num_groups, N] ----
    scales = weight_scale.t().contiguous()

    return weight_int8, scales, zeros_fp16


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

        if c.weight_type == scalar_types.uint4:
            if c.group_size not in SM70_W4_GROUP_SIZES:
                return False, (
                    f"SM70 W4 supports group_size in {SM70_W4_GROUP_SIZES}, "
                    f"got {c.group_size}"
                )
        elif c.weight_type == scalar_types.uint8:
            if c.group_size not in SM70_W8_GROUP_SIZES:
                return False, (
                    f"SM70 W8 supports group_size in {SM70_W8_GROUP_SIZES}, "
                    f"got {c.group_size}"
                )
        else:
            return False, (
                f"SM70 TurboMind kernel supports uint4/uint8 (asym), "
                f"got {c.weight_type}"
            )

        if not c.zero_points:
            # awq_sm70_prepare / w8a16_sm70a_prepare both require qzeros.
            # Sym variants (uint4b8, uint8b128) would need a different path;
            # not implemented here.
            return False, "SM70 TurboMind kernel requires asymmetric (zero_points)"

        if c.has_g_idx:
            return False, "SM70 TurboMind kernel does not support group indices"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Dispatch on weight_type and call the right prepare op.

        Both branches end with the layer carrying `_awq_sm70_*` attributes
        plus a `_sm70_w_bits` marker; `apply_weights` reads the marker to
        pick the right GEMM op. The compressed-tensors source params are
        freed afterwards (single source of truth on `layer`).
        """
        c = self.config
        if c.weight_type == scalar_types.uint4:
            self._process_w4_after_loading(layer)
        elif c.weight_type == scalar_types.uint8:
            self._process_w8_after_loading(layer)
        else:
            raise AssertionError(
                f"SM70 TurboMind kernel: unexpected weight_type {c.weight_type}"
            )

    def _process_w4_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config
        n = c.partition_weight_shape[1]
        k = c.partition_weight_shape[0]
        assert n % 8 == 0 and k % 8 == 0, (
            f"SM70 W4 path needs K and N divisible by 8, got K={k} N={n}"
        )

        w_packed = getattr(layer, self.w_q_name)
        assert isinstance(w_packed, BasevLLMParameter)
        # compressed-tensors W4 weight_packed: [N, K//8] int32 (8 uint4/int32).
        permute_param_layout_(w_packed, input_dim=1, output_dim=0, packed_dim=1)
        weight_packed = w_packed.data.contiguous()

        w_scale = getattr(layer, self.w_s_name)
        assert isinstance(w_scale, BasevLLMParameter)
        permute_param_layout_(w_scale, input_dim=1, output_dim=0)
        weight_scale = w_scale.data.contiguous()

        assert c.zero_points and self.w_zp_name
        w_zp = getattr(layer, self.w_zp_name)
        assert isinstance(w_zp, BasevLLMParameter)
        # compressed-tensors W4 zp: [N//8, num_groups] int32 (8 uint4/int32).
        permute_param_layout_(w_zp, input_dim=1, output_dim=0, packed_dim=0)
        weight_zero_point = w_zp.data.contiguous()

        qweight, qzeros, scales = _ct_w4_to_classic_awq(
            weight_packed, weight_zero_point, weight_scale
        )
        tm_weight, tm_scales, meta = ops.awq_sm70_prepare(
            qweight, scales, qzeros, c.group_size, False
        )
        self._install_prepared(layer, tm_weight, tm_scales, meta, n,
                               w_bits=4, device=device)

    def _process_w8_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config
        n = c.partition_weight_shape[1]
        k = c.partition_weight_shape[0]
        # W8 packs 4 values per int32 (along K for weight, along N for zp).
        # Kernel registry (sm70_884_u8a.cu) further requires N % 4 == 0 for
        # the int8→half conversion and N % 8 for SIMT tile alignment.
        assert n % 8 == 0 and k % 8 == 0, (
            f"SM70 W8 path needs K and N divisible by 8, got K={k} N={n}"
        )

        w_packed = getattr(layer, self.w_q_name)
        assert isinstance(w_packed, BasevLLMParameter)
        # compressed-tensors W8 weight_packed: [N, K//4] int32 (4 uint8/int32).
        permute_param_layout_(w_packed, input_dim=1, output_dim=0, packed_dim=1)
        weight_packed = w_packed.data.contiguous()

        w_scale = getattr(layer, self.w_s_name)
        assert isinstance(w_scale, BasevLLMParameter)
        permute_param_layout_(w_scale, input_dim=1, output_dim=0)
        weight_scale = w_scale.data.contiguous()

        assert c.zero_points and self.w_zp_name
        w_zp = getattr(layer, self.w_zp_name)
        assert isinstance(w_zp, BasevLLMParameter)
        # compressed-tensors W8 zp: [N//4, num_groups] int32 (4 uint8/int32).
        permute_param_layout_(w_zp, input_dim=1, output_dim=0, packed_dim=0)
        weight_zero_point = w_zp.data.contiguous()

        weight_int8, scales, zeros = _ct_w8_unpack(
            weight_packed, weight_zero_point, weight_scale
        )
        tm_weight, tm_scales, meta = ops.w8a16_sm70a_prepare(
            weight_int8, scales, zeros, c.group_size
        )
        self._install_prepared(layer, tm_weight, tm_scales, meta, n,
                               w_bits=8, device=device)

    def _install_prepared(self,
                          layer: torch.nn.Module,
                          tm_weight: torch.Tensor,
                          tm_scales: torch.Tensor,
                          meta: torch.Tensor,
                          n: int,
                          w_bits: int,
                          device: torch.device) -> None:
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
        layer._sm70_w_bits = w_bits
        layer._awq_sm70_prepared = True

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
        reshaped_x = x.reshape(-1, x.shape[-1])
        n_orig = layer._awq_sm70_n_orig
        w_bits = getattr(layer, "_sm70_w_bits", 4)

        if w_bits == 4:
            # Kernel-native AWQ-W4 via ops.awq_gemm_sm70 (Config_U4_g).
            out = ops.awq_gemm_sm70(
                reshaped_x,
                layer._awq_sm70_weight,
                layer._awq_sm70_scales,
                self.config.group_size,
                layer._awq_sm70_k_ld,
                layer._awq_sm70_q_ld,
            )
        else:
            # Kernel-native W8A16 asym via ops.w8a16_sm70a_gemm_out (Config_U8_a).
            # Allocate output explicitly — the W8 path only exposes the
            # `_out` form; matches W4's per-call output allocation behaviour.
            out = torch.empty(
                (reshaped_x.shape[0], n_orig),
                dtype=reshaped_x.dtype,
                device=reshaped_x.device,
            )
            ops.w8a16_sm70a_gemm_out(
                out,
                reshaped_x,
                layer._awq_sm70_weight,
                layer._awq_sm70_scales,
                self.config.group_size,
                layer._awq_sm70_k_ld,
                layer._awq_sm70_q_ld,
                n_orig,
                False,
            )
        return out[:, :n_orig].reshape(x.shape[:-1] + (n_orig,))
