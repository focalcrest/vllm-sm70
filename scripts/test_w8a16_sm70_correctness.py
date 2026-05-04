"""Test W8A16 SM70 correctness: compare INT8 dequantized GEMM vs FP16 reference."""
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops


def quantize_grouped(weight_fp16, group_size=128):
    """Quantize FP16 [N, K] to int8 + per-group FP16 scales."""
    n, k = weight_fp16.shape
    num_groups = k // group_size
    weight_2d = weight_fp16.reshape(n, num_groups, group_size)
    max_abs = weight_2d.abs().amax(dim=2, keepdim=True).clamp(min=1e-5)
    scale = (max_abs / 127.0).squeeze(2).to(torch.float16)  # [N, num_groups]
    weight_q = (weight_2d / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
    weight_int8 = weight_q.reshape(n, k)
    return weight_int8, scale.t()  # [num_groups, N]


def test_w8a16_correctness():
    torch.manual_seed(42)
    device = torch.device("cuda")
    group_size = 128

    test_cases = [
        (1, 4096, 4096),   # decode: M=1
        (4, 4096, 4096),   # small batch
        (8, 4096, 4096),   # medium batch
        (1, 7168, 4096),   # Qwen3-like gate
        (1, 4096, 7168),   # Qwen3-like up
    ]

    for m, n, k in test_cases:
        weight_fp16 = torch.randn(n, k, dtype=torch.float16, device=device)
        x = torch.randn(m, k, dtype=torch.float16, device=device)

        # FP16 reference
        ref_out = torch.mm(x, weight_fp16.t())

        # W8A16 path
        weight_i8, scales = quantize_grouped(weight_fp16, group_size)

        tm_weight, tm_scales, meta = ops.w8a16_sm70_prepare(
            weight_i8, scales, group_size
        )
        w_ld = int(meta[0])
        s_ld = int(meta[1])

        out = torch.empty(m, n, dtype=torch.float16, device=device)
        ops.w8a16_sm70_gemm_out(
            out, x, tm_weight, tm_scales, group_size, w_ld, s_ld, False
        )

        cos_sim = F.cosine_similarity(
            ref_out.flatten().unsqueeze(0),
            out.flatten().unsqueeze(0),
        ).item()
        max_err = (ref_out - out).abs().max().item()
        mean_err = (ref_out - out).abs().mean().item()
        ref_norm = ref_out.abs().mean().item()

        status = "PASS" if cos_sim > 0.99 else "FAIL"
        print(f"[{status}] M={m:3d} N={n:5d} K={k:5d} | "
              f"cos_sim={cos_sim:.6f} max_err={max_err:.4f} "
              f"mean_err={mean_err:.4f} ref_norm={ref_norm:.4f}")


if __name__ == "__main__":
    test_w8a16_correctness()
