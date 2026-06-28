"""BFLA mask-builder (arXiv 2605.12193 Stage-1 + Stage-2), SM70-safe pure-torch.

Builds the MInference-style vertical-block sparse pattern consumed by
vllm_flash_attn.sparse_attn_func: per (query-head, query-row-block) a list of KV
block start-offsets (block_count / block_offset). Column/slash path unused
(local band + sink expressed as blocks).

Stage 1: mean-pool Q into query-row-blocks (size block_m) and K into KV blocks
(size block_n); block-importance = softmax(pooled Q·Kᵀ·scale); keep the smallest
set of KV blocks whose cumulative mass >= gamma (causal-masked).
Stage 2 rescue: sink (block 0), local band (n_local blocks ending at the diagonal),
diagonal block.

Memory-efficient: scores are computed in chunks of `r_chunk` query-row-blocks, so
peak transient memory is O(Hq * r_chunk * C) — bounded regardless of seqlen (needed
for very long context, e.g. 250K).

Granularity MUST match the kernel: block_m = kBlockM (64 for hd128, 32 for hd256),
block_n = kBlockN (64).
"""
import torch


def _pool(x, S, B, n):
    pad = n * B - S
    if pad:
        x = torch.cat([x, x.new_zeros(pad, *x.shape[1:])], 0)
    return x.view(n, B, *x.shape[1:]).mean(1)


@torch.no_grad()
def build_bfla_block_mask(q, k, *, block_m, block_n=64, gamma=0.95, n_local=4,
                          causal=True, max_nnz=None, r_chunk=1024):
    """q:[Sq,Hq,D] k:[Sk,Hk,D] fp16 (single sequence). Returns
    (block_count[1,Hq,R] int32, block_offset[1,Hq,R,NNZ] int32,
     column_count[1,Hq,R] int32, column_index[1,Hq,R,1] int32, density)."""
    Sq, Hq, D = q.shape
    Sk, Hk = k.shape[0], k.shape[-2]
    ratio = Hq // Hk
    dev = q.device
    R = (Sq + block_m - 1) // block_m
    C = (Sk + block_n - 1) // block_n
    scale = D ** -0.5
    shift = Sk - Sq

    Kb = _pool(k.float(), Sk, block_n, C)            # [C, Hk, D]
    Kb_e = Kb.repeat_interleave(ratio, dim=1)        # [C, Hq, D]
    Qb_all = _pool(q.float(), Sq, block_m, R)        # [R, Hq, D]
    kstart = torch.arange(C, device=dev) * block_n   # [C]
    c_ar = torch.arange(C, device=dev)[None, :]      # [1,C]

    block_count = torch.zeros(Hq, R, dtype=torch.int32, device=dev)
    chunks = []                                      # per r-chunk [Hq, rc, nnz_c] offsets
    kept_total = 0
    causal_total = 0

    for r0 in range(0, R, r_chunk):
        r1 = min(R, r0 + r_chunk)
        rc = r1 - r0
        Qbc = Qb_all[r0:r1]                                          # [rc, Hq, D]
        s = torch.einsum('rhd,chd->hrc', Qbc, Kb_e) * scale         # [Hq, rc, C]
        r_ar = torch.arange(r0, r1, device=dev)[:, None]            # [rc,1]
        qend = (r_ar + 1) * block_m - 1 + shift
        cmask = kstart[None, :] <= qend                            # [rc, C] causal block-allow
        if causal:
            s = s.masked_fill(~cmask[None], float('-inf'))
        p = torch.softmax(s, dim=-1)
        sp, idx = torch.sort(p, dim=-1, descending=True)
        csum = sp.cumsum(-1)
        keep_sorted = csum - sp < gamma
        keep = torch.zeros_like(keep_sorted)
        keep.scatter_(-1, idx, keep_sorted)                        # [Hq, rc, C]

        diag_blk = ((r_ar * block_m + shift) // block_n).clamp_(0, C - 1)  # [rc,1]
        sink = (c_ar == 0).expand(rc, C)
        band = (c_ar <= diag_blk) & (c_ar > diag_blk - n_local)
        diag = (c_ar == diag_blk)
        rescue = sink | band | diag
        if causal:
            rescue = rescue & cmask
        keep = keep | rescue[None]
        if causal:
            keep = keep & cmask[None]

        cnt = keep.sum(-1).to(torch.int32)                         # [Hq, rc]
        block_count[:, r0:r1] = cnt
        nnz_c = int(cnt.max().item())
        if max_nnz:
            nnz_c = min(nnz_c, max_nnz)
        nnz_c = max(nnz_c, 1)
        order = torch.argsort(keep.to(torch.int8), dim=-1, descending=True, stable=True)[..., :nnz_c]
        valid = torch.arange(nnz_c, device=dev)[None, None, :] < cnt[..., None]
        kb = torch.where(valid, order, torch.zeros_like(order))
        kb_sorted, _ = torch.sort(torch.where(valid, kb, torch.full_like(kb, C)), dim=-1)
        kb_sorted = torch.where(kb_sorted >= C, torch.zeros_like(kb_sorted), kb_sorted)
        chunks.append((kb_sorted * block_n).to(torch.int32))       # [Hq, rc, nnz_c]
        kept_total += int(cnt.sum().item())
        causal_total += int(cmask.sum().item()) * Hq if causal else rc * C * Hq
        del s, p, sp, idx, csum, keep, keep_sorted

    nnz = max(t.shape[-1] for t in chunks)
    block_offset = torch.zeros(1, Hq, R, nnz, dtype=torch.int32, device=dev)
    r0 = 0
    for t in chunks:
        rc = t.shape[1]
        block_offset[0, :, r0:r0 + rc, :t.shape[-1]] = t
        r0 += rc
    column_count = torch.zeros(1, Hq, R, dtype=torch.int32, device=dev)
    column_index = torch.zeros(1, Hq, R, 1, dtype=torch.int32, device=dev)
    density = kept_total / max(1, causal_total)
    return block_count[None], block_offset, column_count, column_index, density
