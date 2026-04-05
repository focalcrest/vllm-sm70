# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Attention V100/SM70 backend.

This backend keeps the strict fallback behavior from the 1cat prototype:
- prefill uses the optional dense flash-attn-v100 op only for no-prefix cases,
- decode uses the optional paged decode op when available,
- unsupported cases fall back to Triton.
"""

from __future__ import annotations

import torch
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
_paged_kv_utils = None
_warned_prefill_fallback = False
_warned_feature_fallback = False
_warned_decode_fallback = False
_warned_missing_flash_ops = False
_warned_gqa_fallback = False
_warned_prefill_runtime_fallback = False
_logged_prefill_flash = False
_logged_decode_flash = False


def _get_flash_ops():
    """Lazy-load package-local SM70 flash-attn ops if available."""
    global _flash_attn_func, _flash_attn_decode_paged
    if _flash_attn_func is None or _flash_attn_decode_paged is None:
        try:
            from vllm.vllm_flash_attn_sm70 import (  # type: ignore[attr-defined]
                flash_attn_decode_paged,
                flash_attn_func,
            )
        except ImportError:
            _flash_attn_func = None
            _flash_attn_decode_paged = None
        else:
            _flash_attn_func = flash_attn_func
            _flash_attn_decode_paged = flash_attn_decode_paged
    return _flash_attn_func, _flash_attn_decode_paged


def _get_paged_kv_utils():
    """Lazy-load paged KV extraction CUDA extension."""
    global _paged_kv_utils
    if _paged_kv_utils is None:
        try:
            import paged_kv_utils

            _paged_kv_utils = paged_kv_utils
        except ImportError:
            _paged_kv_utils = None
    return _paged_kv_utils


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


def _build_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    """Build a cu_seqlens tensor from per-sequence lengths."""
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=lengths.device),
            torch.cumsum(lengths.to(torch.int32), dim=0),
        )
    )


def _extract_contiguous_kv_from_paged_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    total_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    paged_kv_utils = _get_paged_kv_utils()

    if isinstance(kv_cache, (list, tuple)):
        key_cache, value_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        elif kv_cache.shape[1] == 2:
            key_cache, value_cache = kv_cache.unbind(1)
        else:
            raise ValueError(
                f"Unexpected KV cache shape {tuple(kv_cache.shape)}; "
                "expected dimension 2 at axis 0 or 1"
            )

    if paged_kv_utils is not None:
        k_cont = paged_kv_utils.paged_to_contiguous(key_cache, block_table, seq_lens)
        v_cont = paged_kv_utils.paged_to_contiguous(value_cache, block_table, seq_lens)
        if total_tokens is None:
            total_tokens = int(seq_lens.sum().item())
        return k_cont[:total_tokens], v_cont[:total_tokens]

    batch_size = block_table.shape[0]
    if total_tokens is None:
        total_tokens = int(seq_lens.sum().item())

    k_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=key_cache.dtype,
        device=key_cache.device,
    )
    v_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=value_cache.dtype,
        device=value_cache.device,
    )

    token_offset = 0
    for batch_idx in range(batch_size):
        seq_len = int(seq_lens[batch_idx].item())
        if seq_len == 0:
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            physical_block_idx = int(block_table[batch_idx, block_idx].item())
            start_token = block_idx * block_size
            end_token = min(start_token + block_size, seq_len)
            n = end_token - start_token

            k_cont[token_offset : token_offset + n] = key_cache[physical_block_idx, :n]
            v_cont[token_offset : token_offset + n] = value_cache[
                physical_block_idx, :n
            ]
            token_offset += n

    return k_cont, v_cont


class FlashAttnSM70MetadataBuilder(TritonAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        attn_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flash_attn_func, self.flash_attn_decode_paged = _get_flash_ops()
        self.use_flash_v100 = self.flash_attn_func is not None
        self.use_flash_v100_decode = self.flash_attn_decode_paged is not None
        self._decode_cache_k: torch.Tensor | None = None
        self._decode_cache_v: torch.Tensor | None = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0

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

        block_size = key_cache.shape[-3]
        total_tokens = int(attn_metadata.seq_lens.sum().item())
        key_contig, value_contig = _extract_contiguous_kv_from_paged_cache(
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            num_kv_heads=key_cache.shape[-2],
            head_dim=key_cache.shape[-1],
            block_size=block_size,
            total_tokens=total_tokens,
        )

        self.flash_attn_func(
            q=query,
            k=key_contig,
            v=value_contig,
            out=out_view,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=_build_cu_seqlens(attn_metadata.seq_lens),
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
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
        global _warned_decode_fallback, _warned_prefill_fallback
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
        is_capturing = query.is_cuda and torch.cuda.is_current_stream_capturing()

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
            if is_capturing:
                if not _warned_prefill_fallback:
                    logger.warning(
                        "FLASH_ATTN_SM70 prefill fallback during CUDA graph capture."
                    )
                    _warned_prefill_fallback = True
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
        return self._flash_v100_decode(query, key, value, kv_cache, attn_metadata, output)

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
    ) -> torch.Tensor:
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if query.shape[0] == 0:
            return output

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)

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

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(7, 0)
