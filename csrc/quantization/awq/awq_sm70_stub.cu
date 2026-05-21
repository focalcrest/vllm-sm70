#include <torch/all.h>
#include <string>
#include <vector>

namespace {
[[noreturn]] void awq_sm70_unavailable() {
  TORCH_CHECK(
      false,
      "SM70 AWQ TurboMind kernels are unavailable because lmdeploy sources "
      "were not found during build. This build supports FP16/non-AWQ paths "
      "only.");
}
}  // namespace

#ifndef USE_ROCM

std::vector<torch::Tensor> awq_sm70_prepare(torch::Tensor,
                                            torch::Tensor,
                                            torch::Tensor,
                                            int64_t,
                                            bool) {
  awq_sm70_unavailable();
}

std::vector<torch::Tensor> sm70_f16_prepare(torch::Tensor) {
  awq_sm70_unavailable();
}

torch::Tensor awq_gemm_sm70(torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            int64_t,
                            int64_t,
                            int64_t) {
  awq_sm70_unavailable();
}

torch::Tensor sm70_f16_gemm(torch::Tensor, torch::Tensor) {
  awq_sm70_unavailable();
}

void awq_gemm_sm70_out(torch::Tensor,
                       torch::Tensor,
                       torch::Tensor,
                       torch::Tensor,
                       int64_t,
                       int64_t,
                       int64_t,
                       bool) {
  awq_sm70_unavailable();
}

void sm70_f16_gemm_out(torch::Tensor,
                       torch::Tensor,
                       torch::Tensor,
                       int64_t,
                       bool) {
  awq_sm70_unavailable();
}

void sm70_f16_gate_mul_out(torch::Tensor, torch::Tensor, torch::Tensor) {
  awq_sm70_unavailable();
}

std::vector<torch::Tensor> w8a16_sm70a_prepare(torch::Tensor, torch::Tensor,
                                                torch::Tensor, int64_t) {
  awq_sm70_unavailable();
}

void w8a16_sm70a_gemm_out(torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, int64_t, int64_t, int64_t, int64_t,
                          bool) {
  awq_sm70_unavailable();
}

int64_t sm70_gemm_import_cache(torch::Tensor, const std::string&) {
  awq_sm70_unavailable();
}

int64_t sm70_gemm_export_cache(torch::Tensor, const std::string&) {
  awq_sm70_unavailable();
}

std::vector<torch::Tensor> awq_moe_build_strided_ptrs(torch::Tensor,
                                                       torch::Tensor,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t) {
  awq_sm70_unavailable();
}

void awq_moe_single_token_compact_prepare(torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor) {
  awq_sm70_unavailable();
}

void awq_moe_single_token_sm70_out(torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   torch::Tensor,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t,
                                   int64_t) {
  awq_sm70_unavailable();
}

torch::Tensor awq_moe_gemm_sm70(torch::Tensor,
                                torch::Tensor,
                                torch::Tensor,
                                torch::Tensor,
                                int64_t,
                                int64_t,
                                int64_t,
                                int64_t) {
  awq_sm70_unavailable();
}

void awq_moe_gemm_sm70_out(torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           torch::Tensor,
                           int64_t,
                           int64_t,
                           int64_t,
                           int64_t,
                           bool) {
  awq_sm70_unavailable();
}

#endif  // USE_ROCM
