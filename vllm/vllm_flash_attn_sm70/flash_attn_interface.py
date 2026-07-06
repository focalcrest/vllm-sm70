# Copyright (c) 2023, Tri Dao.

import os
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
# Use relative import to support build-from-source installation in vLLM

from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401

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
    """Return num_splits override for FA decode-path on SM70.

    Default 4 picked from a sweep on Qwen3.6-27B-W8A16 V100 TP=8
    cudagraph (worklog 2026-05-09-fa-num-splits-tuning.md): values 2/4
    give +2.7% over FA's internal auto, 4 is best, 8/16/32 fall off
    slightly. Override with VLLM_SM70_FLASH_ATTN_DECODE_NUM_SPLITS or
    SM70_FLASH_ATTN_DECODE_NUM_SPLITS to test alternative settings.
    Set to 0 to fall back to FA's heuristic.
    """
    raw = os.environ.get("SM70_FLASH_ATTN_DECODE_NUM_SPLITS")
    if raw is None:
        raw = os.environ.get("VLLM_SM70_FLASH_ATTN_DECODE_NUM_SPLITS")
    if raw is None:
        return 4
    try:
        return max(0, int(raw))
    except ValueError:
        return 4


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
    # Hybrid TQ + raw FP16 K/V for continuation prefill
    k_raw=None,
    v_raw=None,
    tq_cached_lens=None,
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
        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
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
            k_raw,
            v_raw,
            tq_cached_lens,
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

    This is a direct wrapper over the native _vllm_fa2_C.fwd_kvcache op.
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

    out_tensors = torch.ops._vllm_fa2_C.fwd_kvcache(
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


# ---------------------------------------------------------------------------
# Two-stage partitioned decode (ported from 1Cat-vLLM flash-attention-v100)
# ---------------------------------------------------------------------------

_partitioned_decode_mod = None
_partitioned_decode_workspace_cache: dict = {}

DEFAULT_PARTITION_SIZE = 256
LONG_CONTEXT_PARTITION_SIZE = 512
LONG_CONTEXT_PARTITION_THRESHOLD = 20480


def _get_partitioned_decode_mod():
    """Load the two-stage partitioned decode CUDA extension for SM70.

    Prefers a prebuilt .so (compiled ahead-of-time on a host with nvcc — see
    scripts/build_sm70_partitioned_decode.py, run during the wheel build) and
    falls back to JIT-compiling from source on first use. The JIT path needs
    nvcc at runtime, which the vllm-sm70-docker image intentionally omits
    (nvidia/cuda:*-runtime, not *-devel); without the prebuilt .so it fails
    silently there and every decode call falls back to the slower FA2 path.
    """
    global _partitioned_decode_mod
    if _partitioned_decode_mod is not None:
        return _partitioned_decode_mod

    import logging
    logger = logging.getLogger(__name__)

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
    sources = [
        os.path.join(csrc_dir, "decode_paged_api.cpp"),
        os.path.join(csrc_dir, "flash_decode_paged.cu"),
    ]
    for src in sources:
        if not os.path.exists(src):
            logger.warning("Partitioned decode source not found: %s", src)
            return None

    prebuilt_so = os.path.join(csrc_dir, "flash_decode_paged_sm70.so")
    if os.path.exists(prebuilt_so):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "flash_decode_paged_sm70", prebuilt_so
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _partitioned_decode_mod = mod
            logger.info("Loaded prebuilt partitioned decode kernel for SM70.")
            return _partitioned_decode_mod
        except Exception as e:
            logger.warning(
                "Failed to load prebuilt partitioned decode kernel (%s: %s); "
                "falling back to JIT compile.", type(e).__name__, e,
            )

    try:
        from torch.utils.cpp_extension import load
        logger.info("JIT-compiling partitioned decode kernel for SM70...")
        _partitioned_decode_mod = load(
            name="flash_decode_paged_sm70",
            sources=sources,
            extra_include_paths=[csrc_dir],
            extra_cuda_cflags=[
                "-gencode", "arch=compute_70,code=sm_70",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-O3",
            ],
            verbose=False,
        )
        logger.info("Partitioned decode kernel compiled successfully.")
        return _partitioned_decode_mod
    except Exception as e:
        logger.warning("Failed to compile partitioned decode kernel: %s", e)
        return None


def _get_partition_size(max_seq_capacity: int) -> int:
    raw = os.environ.get("VLLM_SM70_DECODE_PARTITION_SIZE")
    if raw is not None:
        value = int(raw)
        if value in (256, 512, 1024):
            return value
    if max_seq_capacity > LONG_CONTEXT_PARTITION_THRESHOLD:
        return LONG_CONTEXT_PARTITION_SIZE
    return DEFAULT_PARTITION_SIZE


def _get_partitioned_workspace(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    max_seq_capacity: int = 0,
):
    batch_capacity = block_table.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    derived_capacity = block_table.shape[1] * k_cache.shape[1]
    # Use the larger of the override and derived value so the workspace
    # (and therefore the CUDA grid) is big enough for any future seq_len.
    # Critical for cudagraph: the grid is frozen at capture time.
    max_seq_capacity = max(max_seq_capacity, derived_capacity)
    partition_size = _get_partition_size(max_seq_capacity)
    max_num_partitions = (max_seq_capacity + partition_size - 1) // partition_size
    device_index = q.device.index if q.device.index is not None else -1
    key = (device_index, batch_capacity, num_heads, head_dim,
           max_num_partitions, partition_size)

    workspace = _partitioned_decode_workspace_cache.get(key)
    if workspace is None:
        workspace = (
            torch.empty(
                (batch_capacity, num_heads, max_num_partitions, head_dim),
                dtype=torch.float16, device=q.device,
            ),
            torch.empty(
                (batch_capacity, num_heads, max_num_partitions),
                dtype=torch.float32, device=q.device,
            ),
            torch.empty(
                (batch_capacity, num_heads, max_num_partitions),
                dtype=torch.float32, device=q.device,
            ),
        )
        _partitioned_decode_workspace_cache[key] = workspace

    return workspace, partition_size


def flash_attn_decode_partitioned(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    *,
    out=None,
    softmax_scale=None,
    kv_cache_dtype="auto",
    k_scale=1.0,
    v_scale=1.0,
    max_seq_capacity=0,
):
    """Two-stage partitioned decode attention for SM70 (V100).

    Splits the KV cache into fixed-size partitions (256 or 512 tokens),
    computes attention per partition independently, then reduces via online
    softmax. Much higher SM occupancy than FA2's split-K at long contexts.

    Args:
        q: [B, H, D] query tensor (fp16).
        k_cache: [num_blocks, block_size, H_kv, D] paged key cache.
        v_cache: [num_blocks, block_size, H_kv, D] paged value cache.
        block_table: [B, max_num_blocks] int32 block table.
        seq_lens: [B] int32 sequence lengths.
        out: Optional output tensor [B, H, D].
        softmax_scale: Attention scale (default: 1/sqrt(D)).
    """
    mod = _get_partitioned_decode_mod()
    if mod is None:
        raise RuntimeError("Partitioned decode kernel not available")

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q = maybe_contiguous(q)
    block_table = _maybe_int32_contiguous(block_table)
    seq_lens = _maybe_int32_contiguous(seq_lens)

    # q must be 3D [B, H, D]
    if q.dim() == 4:
        # [B, 1, H, D] → [B, H, D]
        q = q.squeeze(1)
    assert q.dim() == 3, f"Expected q [B, H, D], got {q.shape}"

    if out is not None:
        out = maybe_contiguous(out)
        if out.dim() == 2:
            out = out.view(q.shape[0], q.shape[1], q.shape[2])
        elif out.dim() == 4:
            out = out.squeeze(1)

    (tmp_out, max_logits, exp_sums), partition_size = \
        _get_partitioned_workspace(q, k_cache, block_table, max_seq_capacity)

    out_opt = out  # type: ignore[assignment]
    result = mod.decode_paged_fwd(
        q,
        k_cache,
        v_cache,
        out_opt,
        block_table,
        seq_lens,
        tmp_out,
        max_logits,
        exp_sums,
        softmax_scale,
        partition_size,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
    )

    if out is not None:
        return out
    return result


def flash_attn_prefill_paged(
    q,
    kv_cache,
    block_table,
    seqlen_k,
    *,
    out=None,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    rotary_interleaved=True,
):
    """Paged prefill wrapper for SM70 FlashAttention with TQ KV cache.

    Unlike ``flash_attn_decode_paged`` (which processes a single Q token per
    batch entry), this function handles multi-token Q with paged KV — used for
    continuation prefill where cached tokens sit in the TQ paged KV cache and
    new Q tokens need tiled flash-attention processing.

    Args:
        q: [q_len, num_heads, head_dim] or [1, q_len, num_heads, head_dim].
        kv_cache: Paged KV cache (uint8 for TQ).
        block_table: [1, max_num_pages] int32 block table.
        seqlen_k: [1] int32 — full sequence length (cached + new).
        out: Optional pre-allocated output tensor.
        softmax_scale: Scale for attention. Defaults to 1/sqrt(head_dim).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q = maybe_contiguous(q)
    kv_cache = maybe_contiguous(kv_cache)
    block_table = _maybe_int32_contiguous(block_table)
    seqlen_k = _maybe_int32_contiguous(seqlen_k)
    alibi_slopes = maybe_contiguous(alibi_slopes)

    if q.dim() == 3:
        q = q.unsqueeze(0)  # [q_len, H, D] → [1, q_len, H, D]

    if q.dim() != 4 or q.shape[0] != 1:
        raise ValueError(
            "flash_attn_prefill_paged expects q shaped [q_len, num_heads, head_dim] "
            f"or [1, q_len, num_heads, head_dim]; got {tuple(q.shape)}"
        )

    if out is not None:
        out = maybe_contiguous(out)
        if out.dim() == 3:
            out = out.unsqueeze(0)  # [q_len, H, D] → [1, q_len, H, D]
        if out.dim() != 4 or out.shape[0] != 1:
            raise ValueError(
                "flash_attn_prefill_paged expects out shaped [q_len, num_heads, head_dim] "
                f"or [1, q_len, num_heads, head_dim]; got {tuple(out.shape)}"
            )

    out_tensors = torch.ops._vllm_fa2_C.fwd_kvcache(
        q,
        kv_cache,
        kv_cache,  # v_cache same as k_cache (TQ unified)
        None,  # k_new
        None,  # v_new
        seqlen_k,
        None,  # rotary_cos
        None,  # rotary_sin
        None,  # cache_batch_idx
        None,  # leftpad_k
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        1,  # num_splits=1 → prefill kernel (not split-KV decode)
    )

    if out is not None:
        return out
    return out_tensors[0]
