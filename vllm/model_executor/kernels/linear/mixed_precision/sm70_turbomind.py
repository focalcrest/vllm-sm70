# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

SM70_W8_GROUP_SIZE = 128
SM70_W4_GROUP_SIZES = (32, 64, 128)
SM70_N_PAD = 64

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

        # Accepted weight types (compressed-tensors WNa16 mapping):
        #   uint8b128 — sym W8  (subtract 128, signed int8 path)
        #   uint8     — asym W8 (sym GEMM + Python bias correction)
        #   uint4     — asym W4 (classic AWQ format via ops.awq_sm70_prepare,
        #                kernel-native asym via Config_U4_g)
        is_w8 = c.weight_type in (scalar_types.uint8b128, scalar_types.uint8)
        is_w4 = c.weight_type == scalar_types.uint4
        if not (is_w8 or is_w4):
            return False, (
                f"SM70 TurboMind kernel only supports uint8b128/uint8 (W8) "
                f"and uint4 (asym W4), got {c.weight_type}"
            )

        if is_w8 and c.group_size != SM70_W8_GROUP_SIZE:
            return False, (
                f"SM70 W8 only supports group_size={SM70_W8_GROUP_SIZE}, "
                f"got {c.group_size}"
            )
        if is_w4 and c.group_size not in SM70_W4_GROUP_SIZES:
            return False, (
                f"SM70 W4 supports group_size in {SM70_W4_GROUP_SIZES}, "
                f"got {c.group_size}"
            )
        if is_w4 and not c.zero_points:
            # awq_sm70_prepare requires qzeros. Sym W4 (uint4b8) would need
            # a different code path; not implemented here.
            return False, "SM70 W4 path requires asymmetric (zero_points)"

        if c.has_g_idx:
            return False, "SM70 TurboMind kernel does not support group indices"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        if c.weight_type == scalar_types.uint4:
            self._process_w4_after_loading(layer)
        else:
            self._process_w8_after_loading(layer)

    def _process_w4_after_loading(self, layer: torch.nn.Module) -> None:
        """compressed-tensors W4 → classic AWQ format → awq_sm70_prepare.

        After this returns the layer carries `_awq_sm70_*` tensors (matching
        the convention used by the existing AWQ scheme's SM70 fast-path in
        `awq.py:317`). `apply_weights` dispatches on the presence of these
        tensors vs `_sm70_tm_*` to pick the right gemm op.
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
        layer._awq_sm70_prepared = True

        # Free original packed weights / scales / zeros.
        for name in (self.w_q_name, self.w_s_name, self.w_zp_name):
            if not name:
                continue
            self._transform_param(
                layer, name,
                lambda _: torch.empty(0, dtype=torch.float16, device=device),
            )

    def _process_w8_after_loading(self, layer: torch.nn.Module) -> None:
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

        # Asymmetric path. Compressed-tensors stores `weight_zero_point` as
        # int32-packed uint8b128 with packed_dim along the N axis: shape
        # [N//4, num_groups]. The SM70 HMMA_884 path cannot dequant
        # `s*(q-zp)` natively (see worklog for the V smem-copy assumption
        # that ties it to AWQ-W4 / uint4 weights), so we factor the asym
        # dequant into:
        #
        #     out[m, n] = (sym GEMM)
        #               - sum_g x_per_group[m, g] * (s_g_n * zp_g_n)
        #
        # by storing `zps_scaled[g, n] = s_g_n * zp_g_n` and applying the
        # second term post-GEMM in `apply_weights`. Mathematically exact
        # (modulo fp16 round-off); cost is one extra [M, ng] @ [ng, N] fp16
        # GEMM (<1% of the main GEMM for production shapes).
        zps_scaled: torch.Tensor | None = None
        if c.zero_points and self.w_zp_name:
            w_zp = getattr(layer, self.w_zp_name)
            assert isinstance(w_zp, BasevLLMParameter)
            permute_param_layout_(
                w_zp, input_dim=1, output_dim=0, packed_dim=0
            )
            zp_packed = w_zp.data.contiguous()  # [N//4, num_groups] int32
            num_groups_local = scales.size(0)
            zp_unpacked = torch.zeros(
                n, num_groups_local, dtype=torch.uint8, device=device
            )
            for i, shift in enumerate(shifts):
                zp_unpacked[i::4, :] = (
                    (zp_packed >> shift) & 0xFF
                ).to(torch.uint8)
            zp_signed = (zp_unpacked.to(torch.int16) - 128).to(torch.float16)
            # [N, num_groups] -> [num_groups, N] aligned with scales,
            # then pre-multiply with scales for the correction.
            zps_scaled = (zp_signed * scales.t()).t().contiguous()

        # Pad N to multiple of SM70_N_PAD for TurboMind HMMA kernel
        n_padded = ((n + SM70_N_PAD - 1) // SM70_N_PAD) * SM70_N_PAD
        if n_padded != n:
            weight_int8 = torch.nn.functional.pad(
                weight_int8, (0, 0, 0, n_padded - n)
            )
            scales = torch.nn.functional.pad(scales, (0, n_padded - n))
            if zps_scaled is not None:
                # Pad zps_scaled to n_padded; the padded N rows are sliced
                # off in apply_weights so any value works, 0 is cheapest.
                zps_scaled = torch.nn.functional.pad(
                    zps_scaled, (0, n_padded - n)
                )

        # Prepare weights for TurboMind HMMA_884 kernel (sym GEMM)
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

        # If asym, store pre-multiplied scale*zp [num_groups, n_padded] fp16
        # for the post-GEMM correction in `apply_weights`.
        if zps_scaled is not None:
            layer.register_parameter(
                "_sm70_zps_scaled",
                torch.nn.Parameter(zps_scaled, requires_grad=False),
            )

        # Free original packed weights, scales, and (if asym) zeros.
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
        if c.zero_points and self.w_zp_name:
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda _: torch.empty(0, dtype=torch.float16, device=device),
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # W4 path: kernel-native AWQ via ops.awq_gemm_sm70 (Config_U4_g).
        if getattr(layer, "_awq_sm70_prepared", False):
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

        # W8 path: sym GEMM (+ optional Python bias correction for asym).
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

        # Asymmetric bias correction:
        #   asym_out[m, n] = sym_out[m, n]
        #                  - sum_g x_per_group[m, g] * (s_g_n * zp_g_n)
        # x_per_group is the per-group sum of x along K. The correction is
        # a small [M, ng] @ [ng, n_padded] fp16 GEMM, <1% of main GEMM cost.
        zps_scaled = getattr(layer, "_sm70_zps_scaled", None)
        if zps_scaled is not None:
            gs = self.config.group_size
            m, k = reshaped_x.shape
            xg = reshaped_x.reshape(m, k // gs, gs).sum(dim=2)
            out.sub_(xg @ zps_scaled)

        return out[:, :n_orig].reshape(x.shape[:-1] + (n_orig,))
