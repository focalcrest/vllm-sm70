# Copyright (c) 2023, Tri Dao.

import os
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
# Use relative import to support build-from-source installation in vLLM

from . import _vllm_fa2_sm70_C  # noqa: F401

# isort: on

DEFAULT_FA_VERSION = 2

def _is_fa2_supported(device = None) -> Tuple[bool, Optional[str]]:
    return True, None
    
def _is_fa3_supported(device = None) -> Tuple[bool, Optional[str]]:
    return False, "FA3 is not supported"

def _is_fa4_supported(device = None) -> Tuple[bool, Optional[str]]:
    return False, "FA4 is not supported"

def is_fa_version_supported(fa_version: int, device = None) -> bool:
    if fa_version == 2:
        return True
    else:
        return False

def fa_version_unsupported_reason(fa_version: int, device = None) \
    -> Optional[str]: return None

#
#  For vLLM we only care about `flash_attn_varlen_func` and 
#   `flash_attn_with_kvcache` so we only maintain wrappers for these two.
#


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _maybe_int32_contiguous(x):
    if x is None:
        return None
    if x.dtype != torch.int32:
        x = x.to(dtype=torch.int32)
    return x.contiguous() if not x.is_contiguous() else x


def _decode_num_splits() -> int:
    raw = os.environ.get("SM70_FLASH_ATTN_DECODE_NUM_SPLITS")
    if raw is None:
        raw = os.environ.get("VLLM_SM70_FLASH_ATTN_DECODE_NUM_SPLITS")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _prefill_num_splits(max_seqlen_k: int, num_heads: int) -> int:
    """Override num_splits for prefill to improve SM utilization on V100.

    With few heads (e.g. 3 at TP8), the default split-KV path runs with
    Split=false, leaving most SMs idle. Setting num_splits > 1 enables
    true KV splitting for better parallelism at long sequences.
    """
    raw = os.environ.get("VLLM_SM70_FLASH_ATTN_PREFILL_NUM_SPLITS")
    if raw is not None:
        return max(0, int(raw))
    return 0

# NOTE only used in FA3
def get_scheduler_metadata(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    has_softcap=False,
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    sm_margin=0,     # Can be tuned if some SMs are used for communication
):
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = torch.ops._vllm_fa3_C.get_scheduler_metadata(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim, headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0], window_size[1],
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )

    return scheduler_metadata


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None, # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[List[int]] = None,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert block_table is None or seqused_k is not None, \
        "seqused_k must be provided if block_table is provided"
    
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # custom op does not support non-tuple input
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    
    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    
    if fa_version == 2:
        if scheduler_metadata is not None and q_descale is not None \
            and k_descale is not None and v_descale is not None:
                raise NotImplementedError(
                    "FA2 does not support scheduler_metadata, q_descale, "
                    "k_descale, v_descale"
                )
        if s_aux is not None:
            raise NotImplementedError("FA2 does not support s_aux")
        # Allow split-KV for prefill when env var is set
        if num_splits <= 1 and max_seqlen_q > 1:
            num_splits = _prefill_num_splits(max_seqlen_k, q.shape[1])
        out, softmax_lse = torch.ops._vllm_fa2_sm70_C.varlen_fwd(
            q, k, v,
            out,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp 
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            return_softmax_lse and dropout_p > 0,
            num_splits,
            None,
        )
    elif fa_version == 3:
        assert alibi_slopes is None, "Alibi is not supported in FA3"
        out, softmax_lse, _, _ = torch.ops._vllm_fa3_C.fwd(
            q, k, v,
            None, None,       # k_new, v_new
            q_v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,     # cu_seqlens_k
            None,             # cu_seqlens_k_new
            None, seqused_k,  # seqused_q, seqused_k
            max_seqlen_q, max_seqlen_k,
            block_table,
            None,             # kv_batch_idx
            None,             # leftpad_k
            None, None, None, # rotary_cos, rotary_sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal,
            real_window_size[0], real_window_size[1],
            softcap,
            True,             # rotary_interleaved
            scheduler_metadata,
            num_splits,
            None,             # pack_gqa
            0,                # sm_margin
            s_aux,            # s_aux
            cp_world_size,
            cp_rank,
            cp_tot_seqused_k,
        )
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")
    return (out, softmax_lse) if return_softmax_lse else out


def flash_attn_decode_paged(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    *,
    out=None,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    rotary_interleaved=True,
    num_splits=0,
):
    """Paged decode wrapper for SM70 FlashAttention.

    This is a direct wrapper over the native _vllm_fa2_sm70_C.fwd_kvcache op.
    It expects the current vLLM paged KV cache layout and block table directly,
    without materializing a contiguous intermediate buffer.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q = maybe_contiguous(q)
    k_cache = maybe_contiguous(k_cache)
    v_cache = maybe_contiguous(v_cache)
    block_table = _maybe_int32_contiguous(block_table)
    seq_lens = _maybe_int32_contiguous(seq_lens)
    alibi_slopes = maybe_contiguous(alibi_slopes)

    if q.dim() == 3:
        q = q.unsqueeze(1)

    if q.dim() != 4 or q.shape[1] != 1:
        raise ValueError(
            "flash_attn_decode_paged expects q shaped [batch, 1, num_heads, head_dim] "
            f"(or [batch, num_heads, head_dim]); got {tuple(q.shape)}"
        )

    if out is not None:
        out = maybe_contiguous(out)
        if out.dim() == 2:
            out = out.view(q.shape[0], q.shape[1], q.shape[2], q.shape[3])
        elif out.dim() == 3:
            out = out.unsqueeze(1)
        if out.dim() != 4:
            raise ValueError(
                "flash_attn_decode_paged expects out shaped [batch, hidden], "
                "[batch, num_heads, head_dim], or [batch, 1, num_heads, head_dim]; "
                f"got {tuple(out.shape)}"
            )

    out_tensors = torch.ops._vllm_fa2_sm70_C.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        None,
        None,
        seq_lens,
        None,
        None,
        None,
        None,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        _decode_num_splits() if num_splits <= 0 else num_splits,
    )

    # The custom op mutates `out` in-place when provided. Keep the return value
    # simple for the backend call-site.
    if out is not None:
        return out
    return out_tensors[0]
