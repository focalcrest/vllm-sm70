# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Attention V100/SM70 backend.

This backend keeps the strict fallback behavior from the 1cat prototype:
- prefill uses the optional dense flash-attn-v100 op only for no-prefix cases,
- decode uses the optional paged decode op when available,
- unsupported cases fall back to Triton.
"""

from __future__ import annotations

import os

import torch
from typing import ClassVar

from vllm.config.cache import CacheDType
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from typing_extensions import override

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

logger = init_logger(__name__)

# Lazy imports: only resolve optional CUDA extensions when needed.
_flash_attn_func = None
_flash_attn_decode_paged = None
_flash_attn_decode_partitioned = None
_flash_attn_prefill_paged = None
_warned_feature_fallback = False
_warned_decode_fallback = False
_warned_decode_runtime_fallback = False
_warned_missing_flash_ops = False
_warned_gqa_fallback = False
_warned_prefill_runtime_fallback = False
_logged_prefill_flash = False
_logged_decode_flash = False


def _get_flash_ops():
    """Lazy-load package-local SM70 flash-attn ops if available."""
    global _flash_attn_func, _flash_attn_decode_paged, _flash_attn_decode_partitioned, _flash_attn_prefill_paged
    if _flash_attn_func is None or _flash_attn_decode_paged is None:
        try:
            from vllm.vllm_flash_attn_sm70 import (  # type: ignore[attr-defined]
                flash_attn_decode_paged,
                flash_attn_decode_partitioned,
                flash_attn_func,
                flash_attn_prefill_paged,
            )
        except ImportError:
            _flash_attn_func = None
            _flash_attn_decode_paged = None
            _flash_attn_decode_partitioned = None
            _flash_attn_prefill_paged = None
        else:
            _flash_attn_func = flash_attn_func
            _flash_attn_decode_paged = flash_attn_decode_paged
            _flash_attn_decode_partitioned = flash_attn_decode_partitioned
            _flash_attn_prefill_paged = flash_attn_prefill_paged
    return _flash_attn_func, _flash_attn_decode_paged, _flash_attn_decode_partitioned, _flash_attn_prefill_paged


def _has_prefix_context(attn_metadata: TritonAttentionMetadata) -> bool:
    query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
    return not torch.equal(query_lens, attn_metadata.seq_lens)


def _is_cascade_supported(attn_metadata: TritonAttentionMetadata) -> bool:
    return (
        attn_metadata.common_prefix_len > 0
        and attn_metadata.alibi_slopes is None
        and attn_metadata.logits_soft_cap == 0
        and attn_metadata.sinks is None
        and attn_metadata.sliding_window == (-1, -1)
    )


class FlashAttnSM70MetadataBuilder(TritonAttentionMetadataBuilder):
    # UNIFORM_BATCH (not UNIFORM_SINGLE_TOKEN_DECODE) so the engine does NOT
    # downgrade FULL_AND_PIECEWISE -> PIECEWISE when speculative decoding is
    # active. Spec-decode verify steps have query_len = 1 + num_speculative
    # tokens (a uniform multi-token batch), which this backend captures
    # correctly into a FULL cudagraph (validated lossless; see worklog
    # 2026-06-25-dflash-cudagraph-uniform-batch-and-anbeeld-repro). Pairs with
    # the removal of the is_capturing prefill->Triton fallback in forward()
    # below, so the fast flash_attn_func (not the slow Triton fallback) is what
    # gets captured for the verify step.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        attn_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        # DFlash routes its non-causal draft to this backend; thread the flag
        # through so forward() (and the DFlash per-layer assert) see it.
        # Defaults to True for the target model and all non-DFlash workloads.
        attn_metadata.causal = getattr(common_attn_metadata, "causal", True)
        return attn_metadata

    @staticmethod
    def use_cascade_attention(
        common_prefix_len: int,
        query_lens,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
        dcp_world_size: int,
    ) -> bool:
        del query_lens, num_query_heads, num_kv_heads, num_sms, dcp_world_size
        return (
            common_prefix_len > 0
            and not use_alibi
            and not use_sliding_window
            and not use_local_attention
        )


class FlashAttnSM70Impl(TritonAttentionImpl):
    # Minimum sequence length to use the partitioned decode kernel.
    # Below this threshold, FA2's fwd_kvcache with num_splits=4 is faster.
    _PARTITIONED_DECODE_THRESHOLD = int(
        os.environ.get("VLLM_SM70_PARTITIONED_DECODE_THRESHOLD", "1024")
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (self.flash_attn_func, self.flash_attn_decode_paged,
         self.flash_attn_decode_partitioned,
         self.flash_attn_prefill_paged) = _get_flash_ops()
        self.use_flash_v100 = self.flash_attn_func is not None
        self.use_flash_v100_decode = self.flash_attn_decode_paged is not None
        self.use_flash_v100_prefill_paged = self.flash_attn_prefill_paged is not None
        # Partitioned decode: enabled by default, JIT-compiled on first use.
        # Disable with VLLM_SM70_PARTITIONED_DECODE=0.
        self.use_partitioned_decode = (
            os.environ.get("VLLM_SM70_PARTITIONED_DECODE", "1") != "0"
            and self.flash_attn_decode_partitioned is not None
        )
        self._decode_cache_k: torch.Tensor | None = None
        self._decode_cache_v: torch.Tensor | None = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0
        self._tq_config = None
        if self._is_tq_cache():
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )
            self._tq_config = TurboQuantConfig.from_cache_dtype(
                self.kv_cache_dtype, self.head_size
            )
            # Note: FP8 keys (k8v4) don't need Hadamard rotation.
            # MSE key paths would need self._tq_PiT = _build_hadamard(...).

    def _reset_decode_cache(self) -> None:
        self._decode_cache_k = None
        self._decode_cache_v = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0

    def _ensure_decode_cache_capacity(
        self,
        required_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and self._decode_cache_capacity >= required_len
            and self._decode_cache_k.shape[1] == num_kv_heads
            and self._decode_cache_k.shape[2] == head_dim
            and self._decode_cache_k.dtype == dtype
            and self._decode_cache_k.device == device
        ):
            return

        new_capacity = max(required_len, max(16, self._decode_cache_capacity * 2))
        new_k = torch.empty((new_capacity, num_kv_heads, head_dim), dtype=dtype, device=device)
        new_v = torch.empty((new_capacity, num_kv_heads, head_dim), dtype=dtype, device=device)

        if (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and self._decode_cache_len > 0
        ):
            new_k[: self._decode_cache_len].copy_(self._decode_cache_k[: self._decode_cache_len])
            new_v[: self._decode_cache_len].copy_(self._decode_cache_v[: self._decode_cache_len])

        self._decode_cache_k = new_k
        self._decode_cache_v = new_v
        self._decode_cache_capacity = new_capacity

    def _supports_flash_v100_path(self) -> bool:
        return (
            self.use_flash_v100
            and self.attn_type == AttentionType.DECODER
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self.sliding_window == (-1, -1)
            and not self.kv_cache_dtype.startswith("fp8")
        )

    def _is_tq_cache(self) -> bool:
        return self.kv_cache_dtype.startswith("turboquant_")

    def _ensure_tq_on_device(self, layer: torch.nn.Module, device: torch.device):
        """One-time derivation of TQ buffers for the layer."""
        if hasattr(layer, "_tq_cached"):
            return
        from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
        D = self.head_size
        H = _build_hadamard(D, str(device))
        layer._tq_PiT = H  # H is already orthonormal (H = H^T)
        if self._tq_config.key_fp8:
            midpoints = torch.tensor([], device=device, dtype=torch.float32)
        else:
            n_bits = self._tq_config.key_mse_bits
            levels = 1 << n_bits
            midpoints = (torch.arange(levels - 1, device="cpu", dtype=torch.float32)
                         + 0.5) * (2.0 / levels) - 1.0
            midpoints = midpoints.to(device=device)
        layer._tq_midpoints = midpoints
        layer._tq_cached = True

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if not self._is_tq_cache():
            return super().do_kv_cache_update(
                layer, key, value, kv_cache, slot_mapping
            )
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        self._ensure_tq_on_device(layer, key.device)
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        from vllm.v1.attention.ops.triton_turboquant_store import (
            triton_turboquant_store,
        )
        triton_turboquant_store(
            k,
            v,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            layer._tq_midpoints,
            mse_bits=self._tq_config.key_mse_bits,
            key_packed_size=self._tq_config.key_packed_size,
            value_quant_bits=self._tq_config.effective_value_quant_bits,
            key_fp8=self._tq_config.key_fp8,
        )

    def _flash_v100_cascade(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)

        block_size = key_cache.shape[-3]
        common_prefix_len = attn_metadata.common_prefix_len
        num_common_kv_blocks = common_prefix_len // block_size

        prefix_output, prefix_lse = self.flash_attn_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.cu_prefix_query_lens,
            seqused_k=attn_metadata.prefix_kv_lens,
            max_seqlen_q=num_actual_tokens,
            max_seqlen_k=common_prefix_len,
            softmax_scale=self.scale,
            causal=False,
            block_table=attn_metadata.block_table[:1],
            return_softmax_lse=True,
        )

        suffix_output, suffix_lse = self.flash_attn_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.suffix_kv_lens,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len - common_prefix_len,
            softmax_scale=self.scale,
            causal=True,
            block_table=attn_metadata.block_table[:, num_common_kv_blocks:],
            return_softmax_lse=True,
        )

        merge_attn_states(
            out_view,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
        )
        return output

    def _flash_v100_chunked_prefill(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)

        self.flash_attn_func(
            q=query,
            k=key_cache,
            v=value_cache,
            out=out_view,
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
        )
        return output

    def _flash_v100_bfla_prefill(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """BFLA sparse-prefill (single sequence, long ctx).

        Gathers the sequence's paged KV into a contiguous buffer, builds the
        BFLA block-importance mask (arXiv 2605.12193), and runs the SM70
        block-sparse flash kernel. Gated to single-sequence prefill (the eval
        scenario); multi-seq / chunked / prefix-cache fall through to dense.
        """
        from vllm.vllm_flash_attn.flash_attn_interface import sparse_attn_func
        from vllm.v1.attention.backends.bfla_mask import build_bfla_block_mask

        num_actual = attn_metadata.num_actual_tokens
        q = query[:num_actual]                                  # [Sq, Hq, D]
        Sq, Hq, D = q.shape
        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)
        block_size, Hk = key_cache.shape[1], key_cache.shape[2]
        seq_len = int(attn_metadata.seq_lens[0].item())
        n_blocks = (seq_len + block_size - 1) // block_size
        phys = attn_metadata.block_table[0, :n_blocks]
        # gather paged KV -> contiguous [1, Sk, Hk, D]
        k = key_cache[phys].reshape(-1, Hk, D)[:seq_len].unsqueeze(0).contiguous()
        v = value_cache[phys].reshape(-1, Hk, D)[:seq_len].unsqueeze(0).contiguous()
        qb = q.unsqueeze(0).contiguous()                        # [1, Sq, Hq, D]

        block_m = 32 if D == 256 else 64                        # kBlockM (smem)
        gamma = float(os.environ.get("VLLM_SM70_BFLA_GAMMA", "0.95"))
        n_local = int(os.environ.get("VLLM_SM70_BFLA_NLOCAL", "8"))
        bc, bo, cc, ci, _density = build_bfla_block_mask(
            qb[0], k[0], block_m=block_m, block_n=64,
            gamma=gamma, n_local=n_local, causal=True,
        )
        if os.environ.get("VLLM_SM70_BFLA_DEBUG") == "1":
            logger.info(
                "BFLA prefill seq_len=%d head_dim=%d gamma=%.2f density=%.3f",
                seq_len, D, gamma, _density,
            )
        out = sparse_attn_func(
            qb, k, v, bc, bo, cc, ci,
            softmax_scale=self.scale, causal=True,
        )
        output[:num_actual] = out[0]
        return output

    def _flash_v100_noncausal(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Non-causal paged attention for the DFlash draft block.

        Mirrors _flash_v100_chunked_prefill but bidirectional (causal=False):
        each query token attends to the full (context + block) paged KV. The
        SM70 flash op supports causal=False + block_table (cf. the cascade
        prefix path). Sliding-window draft layers are treated as full
        attention here, which is exact while context < window (short prompts);
        long-context windowing would need the DFlash-SWA work (PR #40898).
        """
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)

        self.flash_attn_func(
            q=query,
            k=key_cache,
            v=value_cache,
            out=out_view,
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=False,
        )
        return output

    @override
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global _logged_decode_flash, _logged_prefill_flash
        global _warned_decode_fallback
        global _warned_feature_fallback, _warned_missing_flash_ops
        global _warned_gqa_fallback, _warned_prefill_runtime_fallback

        if attn_metadata is None:
            assert output is not None
            return output.fill_(0)

        if not self.use_flash_v100 and not _warned_missing_flash_ops:
            logger.warning(
                "FLASH_ATTN_SM70 backend selected, but optional module "
                "'flash_attn_v100' is unavailable. Falling back to Triton."
            )
            _warned_missing_flash_ops = True

        # DFlash routes its non-causal draft block to this backend. Handle it
        # before the _supports_flash_v100_path() gate so the drafter's
        # sliding-window layers (window >= seq for short ctx => full attention)
        # don't bail to the causal-only Triton fallback.
        if self.use_flash_v100 and getattr(attn_metadata, "causal", True) is False:
            if not _logged_prefill_flash:
                logger.info(
                    "FLASH_ATTN_SM70 non-causal (DFlash draft) path active."
                )
                _logged_prefill_flash = True
            try:
                return self._flash_v100_noncausal(
                    query, kv_cache, attn_metadata, output
                )
            except (RuntimeError, ValueError) as e:
                logger.warning(
                    "FLASH_ATTN_SM70 non-causal path failed (%s: %s); "
                    "falling back to Triton.",
                    type(e).__name__,
                    e,
                )
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )

        if not self._supports_flash_v100_path():
            if self.use_flash_v100 and not _warned_feature_fallback:
                logger.warning(
                    "FLASH_ATTN_SM70 fallback to Triton due to unsupported "
                    "attention features (alibi/softcap/sliding window/fp8/etc)."
                )
                _warned_feature_fallback = True
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        is_prefill = attn_metadata.max_query_len > 1

        if is_prefill and self._is_tq_cache():
            # TQ prefill: first-chunk uses raw FP16 K/V with FA2 varlen.
            # Continuation uses FA2 TQ decode kernel against compressed
            # KV cache (per-request, synthetic seq_lens for causal mask).
            if not _logged_prefill_flash:
                logger.info("FLASH_ATTN_SM70 TQ prefill path active.")
                _logged_prefill_flash = True
            self._reset_decode_cache()
            return self._tq_prefill(
                query, key, value, kv_cache, attn_metadata, output
            )

        if is_prefill:
            if query.shape[1] % key.shape[1] != 0:
                if self.use_flash_v100 and not _warned_gqa_fallback:
                    logger.warning(
                        "FLASH_ATTN_SM70 prefill fallback: unsupported head "
                        "layout (q_heads=%d, kv_heads=%d). Using Triton path.",
                        query.shape[1],
                        key.shape[1],
                    )
                    _warned_gqa_fallback = True
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            # NOTE: no is_capturing prefill->Triton fallback here. With
            # _cudagraph_support = UNIFORM_BATCH (above), the spec-decode verify
            # step (query_len = 1 + num_speculative tokens) is captured into a
            # FULL cudagraph; flash_attn_func is cuda-graph-safe on SM70 for this
            # path (validated lossless). Capturing the Triton fallback instead
            # was measured ~8% slower at deep nspec / long context.
            if attn_metadata.common_prefix_len > 0 and _is_cascade_supported(
                attn_metadata
            ):
                if not _logged_prefill_flash:
                    logger.info("FLASH_ATTN_SM70 cascade prefill path active.")
                    _logged_prefill_flash = True
                self._reset_decode_cache()
                try:
                    return self._flash_v100_cascade(
                        query, kv_cache, attn_metadata, output
                    )
                except (RuntimeError, ValueError):
                    if not _warned_prefill_runtime_fallback:
                        logger.warning(
                            "FLASH_ATTN_SM70 cascade prefill op failed at runtime; "
                            "falling back."
                        )
                        _warned_prefill_runtime_fallback = True
                    return super().forward(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        output_scale,
                        output_block_scale,
                    )
            # BFLA sparse-prefill gate (env VLLM_SM70_BFLA=1): single-sequence,
            # long-ctx, head_dim 128/256. Falls back to dense on any failure.
            if (
                os.environ.get("VLLM_SM70_BFLA") == "1"
                and attn_metadata.seq_lens.shape[0] == 1
                and attn_metadata.max_seq_len
                >= int(os.environ.get("VLLM_SM70_BFLA_MIN_LEN", "8192"))
                and query.shape[-1] in (128, 256)
            ):
                self._reset_decode_cache()
                try:
                    if not _logged_prefill_flash:
                        logger.info("FLASH_ATTN_SM70 BFLA sparse prefill active.")
                        _logged_prefill_flash = True
                    return self._flash_v100_bfla_prefill(
                        query, kv_cache, attn_metadata, output
                    )
                except (RuntimeError, ValueError) as e:
                    logger.warning(
                        "FLASH_ATTN_SM70 BFLA prefill failed (%s: %s); "
                        "falling back to dense.", type(e).__name__, e
                    )
            if not _logged_prefill_flash:
                logger.info("FLASH_ATTN_SM70 prefill path active.")
                _logged_prefill_flash = True
            self._reset_decode_cache()
            try:
                return self._flash_v100_chunked_prefill(
                    query, kv_cache, attn_metadata, output
                )
            except (RuntimeError, ValueError):
                if not _warned_prefill_runtime_fallback:
                    logger.warning(
                        "FLASH_ATTN_SM70 prefill op failed at runtime; falling back."
                    )
                    _warned_prefill_runtime_fallback = True
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )

        if not self.use_flash_v100_decode:
            if self.use_flash_v100 and not _warned_decode_fallback:
                logger.warning(
                    "FLASH_ATTN_SM70 decode fallback to Triton: paged decode op "
                    "is unavailable."
                )
                _warned_decode_fallback = True
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if not _logged_decode_flash:
            logger.info("FLASH_ATTN_SM70 decode path active.")
            _logged_decode_flash = True
        try:
            return self._flash_v100_decode(
                query, key, value, kv_cache, attn_metadata, output, layer
            )
        except (RuntimeError, ValueError, IndexError) as e:
            logger.warning(
                "FLASH_ATTN_SM70 decode op failed at runtime (%s: %s); "
                "disabling and falling back to Triton.",
                type(e).__name__, e,
            )
            self.use_flash_v100_decode = False
            if self.use_flash_v100 and not _warned_decode_runtime_fallback:
                _warned_decode_runtime_fallback = True
                _warned_decode_runtime_fallback = True
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

    def _tq_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """TQ prefill: first-chunk uses raw FP16 K/V, continuation uses
        FA2 prefill kernel with paged TQ KV cache (batched varlen)."""
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        # First-chunk fast path: all K/V in batch
        if attn_metadata.max_query_len == attn_metadata.max_seq_len:
            self.flash_attn_func(
                q=query,
                k=key,
                v=value,
                out=out_view,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                causal=True,
            )

            return output

        # Mixed or continuation: separate first-chunk vs continuation requests.
        query_start_loc_cpu = getattr(
            attn_metadata, "query_start_loc_cpu", None
        )
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        seq_lens = attn_metadata.seq_lens
        num_seqs = len(query_start_loc) - 1

        first_indices = []
        cont_indices = []
        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            q_len = end - start
            seq_len = int(seq_lens[i].item())
            if seq_len - q_len <= 0:
                first_indices.append(i)
            else:
                cont_indices.append(i)

        # First-chunk requests: batched varlen with raw FP16 K/V.
        if first_indices:
            q_parts, kv_parts = [], []
            cu_q_first = torch.zeros(
                len(first_indices) + 1, dtype=torch.int32, device=query.device
            )
            for j, i in enumerate(first_indices):
                s = int(query_start_loc[i].item())
                e = int(query_start_loc[i + 1].item())
                q_parts.append(query[s:e])
                kv_parts.append(key[s:e])
                cu_q_first[j + 1] = cu_q_first[j] + (e - s)
            q_first = torch.cat(q_parts, dim=0)
            k_first = torch.cat(kv_parts, dim=0)
            v_first = torch.cat(
                [value[int(query_start_loc[i].item()):int(query_start_loc[i+1].item())]
                 for i in first_indices], dim=0
            )
            max_q_first = int(cu_q_first[-1].item()) // len(first_indices) if first_indices else 1
            max_q_first = max(
                int(query_start_loc[i+1].item()) - int(query_start_loc[i].item())
                for i in first_indices
            )
            out_first = self.flash_attn_func(
                q=q_first, k=k_first, v=v_first,
                cu_seqlens_q=cu_q_first, cu_seqlens_k=cu_q_first,
                max_seqlen_q=max_q_first, max_seqlen_k=max_q_first,
                softmax_scale=self.scale, causal=True,
            )
            offset = 0
            for i in first_indices:
                s = int(query_start_loc[i].item())
                qlen = int(query_start_loc[i + 1].item()) - s
                out_view[s:s + qlen].copy_(out_first[offset:offset + qlen])
                offset += qlen

        # Continuation requests: hybrid TQ + raw FP16 K/V.
        # Cached tokens are read from TQ paged cache (dequantize in kernel).
        # Current-chunk tokens are read from raw FP16 (fast CuTe async copy).
        if cont_indices:
            q_parts, k_parts, v_parts = [], [], []
            cu_q_cont = torch.zeros(
                len(cont_indices) + 1, dtype=torch.int32, device=query.device
            )
            for j, i in enumerate(cont_indices):
                s = int(query_start_loc[i].item())
                e = int(query_start_loc[i + 1].item())
                q_parts.append(query[s:e])
                k_parts.append(key[s:e])
                v_parts.append(value[s:e])
                cu_q_cont[j + 1] = cu_q_cont[j] + (e - s)
            q_cont = torch.cat(q_parts, dim=0)
            k_cont = torch.cat(k_parts, dim=0)
            v_cont = torch.cat(v_parts, dim=0)

            cont_bt = attn_metadata.block_table[cont_indices]
            cont_seq_lens = seq_lens[cont_indices]
            max_q_cont = max(
                int(query_start_loc[i+1].item()) - int(query_start_loc[i].item())
                for i in cont_indices
            )
            max_seq_cont = int(cont_seq_lens.max().item())

            # Per-request cached token count: seq_len - q_len
            q_lens = cu_q_cont[1:] - cu_q_cont[:-1]
            cached_lens = cont_seq_lens.to(device=query.device, dtype=torch.int32) - q_lens

            out_cont = self.flash_attn_func(
                q=q_cont, k=kv_cache, v=kv_cache,
                cu_seqlens_q=cu_q_cont,
                seqused_k=cont_seq_lens,
                block_table=cont_bt,
                max_seqlen_q=max_q_cont,
                max_seqlen_k=max_seq_cont,
                softmax_scale=self.scale,
                causal=True,
                k_raw=k_cont,
                v_raw=v_cont,
                tq_cached_lens=cached_lens,
            )
            offset = 0
            for i in cont_indices:
                s = int(query_start_loc[i].item())
                qlen = int(query_start_loc[i + 1].item()) - s
                out_view[s:s + qlen].copy_(out_cont[offset:offset + qlen])
                offset += qlen

        return output

    def _flash_v100_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        num_seqs = len(query_start_loc) - 1

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            seqlen = end - start
            cu_seqlens = torch.tensor(
                [0, seqlen], dtype=torch.int32, device=query.device
            )
            out_seq = self.flash_attn_func(
                query[start:end],
                key[start:end],
                value[start:end],
                seqlen,
                cu_seqlens,
                seqlen,
                cu_seqlens_k=cu_seqlens,
                softmax_scale=self.scale,
                causal=True,
            )
            out_view[start:end].copy_(out_seq)

        return output

    def _flash_v100_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if query.shape[0] == 0:
            return output

        # TQ decode: two paths selectable via VLLM_TQ_TRITON_DECODE env var
        #   "1" (default) = Triton FP32 decode (better quality, ~1078 think tokens)
        #   "0" = FA2 HMMA decode (FP16 arithmetic, ~1150 think tokens)
        if self._is_tq_cache():
            use_triton = os.environ.get("VLLM_TQ_TRITON_DECODE", "1") == "1"
            if use_triton:
                from vllm.v1.attention.ops.triton_turboquant_decode import (
                    triton_turboquant_decode_attention,
                )
                assert layer is not None
                self._ensure_tq_on_device(layer, query.device)
                N = num_actual_tokens
                q = query.view(N, self.num_heads, self.head_size)
                result = triton_turboquant_decode_attention(
                    query=q,
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    Pi=layer._tq_PiT,
                    centroids=layer._tq_centroids,
                    scale=self.scale,
                    mse_bits=self._tq_config.key_mse_bits,
                    key_packed_size=self._tq_config.key_packed_size,
                    value_quant_bits=self._tq_config.effective_value_quant_bits,
                    key_fp8=self._tq_config.key_fp8,
                    norm_correction=self._tq_config.norm_correction,
                    PiT=layer._tq_PiT,
                    mid_o_buf=getattr(layer, "_tq_mid_o_buf", None),
                    output_buf=getattr(layer, "_tq_output_buf", None),
                    lse_buf=getattr(layer, "_tq_lse_buf", None),
                    buf_holder=layer,
                    max_num_kv_splits=32,
                )
                out_view.view(N, -1).copy_(
                    result.reshape(N, -1).to(out_view.dtype)
                )
                return output
            else:
                key_cache = kv_cache
                value_cache = kv_cache
        elif kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)

        # Use partitioned decode unconditionally when enabled — the threshold
        # check was removed because cudagraph FULL capture freezes the branch
        # taken during capture (short seq_len → FA2), preventing the
        # partitioned path from ever running at replay time.  The partitioned
        # kernel handles short contexts fine (1 partition ≈ FA2 cost).
        #
        # max_seq_capacity is derived from the block_table column count
        # (which is fixed at cdiv(max_model_len, block_size) in vLLM v1)
        # times the block size.  Passing it explicitly ensures the workspace
        # and CUDA grid are always sized for the worst case, so cudagraph
        # captures a grid that works for any seq_len.
        if self.use_partitioned_decode:
            max_seq_capacity = (attn_metadata.block_table.shape[1]
                                * key_cache.shape[1])
            try:
                self.flash_attn_decode_partitioned(
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    softmax_scale=self.scale,
                    out=out_view,
                    max_seq_capacity=max_seq_capacity,
                )
                return output
            except (RuntimeError, ValueError) as e:
                logger.warning(
                    "Partitioned decode failed (%s: %s); "
                    "falling back to FA2 fwd_kvcache.",
                    type(e).__name__, e,
                )
                self.use_partitioned_decode = False

        self.flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            softmax_scale=self.scale,
            out=out_view,
        )
        return output


class FlashAttnSM70Backend(TritonAttentionBackend):
    """Flash Attention backend for SM70."""

    forward_includes_kv_cache_update: bool = False

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "turboquant_k8v4",
    ]

    @staticmethod
    def get_impl_cls():
        return FlashAttnSM70Impl

    @staticmethod
    def get_builder_cls():
        return FlashAttnSM70MetadataBuilder

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_SM70"

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            tq_config = TurboQuantConfig.from_cache_dtype(
                cache_dtype_str, head_size
            )
            return (
                num_blocks,
                block_size,
                num_kv_heads,
                tq_config.slot_size_aligned,
            )
        return TritonAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str.startswith("turboquant_"):
            return (0, 1, 2, 3)
        return TritonAttentionBackend.get_kv_cache_stride_order(
            include_num_layers_dimension
        )

    @classmethod
    def supports_non_causal(cls) -> bool:
        # The SM70 flash op supports causal=False over paged KV (see the
        # cascade-prefix path); FlashAttnSM70Impl.forward routes non-causal
        # metadata (e.g. the DFlash draft) through _flash_v100_noncausal. This
        # lets the selector pick FLASH_ATTN_SM70 for DFlash's non-causal drafter
        # instead of FlexAttention (which exceeds SM70's 96KB smem at headdim128).
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(7, 0)
