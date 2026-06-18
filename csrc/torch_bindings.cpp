// Provides torch::Tensor for ops.h (previously included transitively via
// cache.h, which is no longer included here after cache ops moved to
// _C_stable_libtorch).
#include <torch/all.h>
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"
#include <torch/library.h>
#include <torch/version.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops
  //

  ops.def(
      "persistent_masked_m_silu_mul_quant(Tensor input, Tensor counts, Tensor! "
      "y_q, Tensor! y_s,"
      "bool use_ue8m0) -> ()");
  ops.impl("persistent_masked_m_silu_mul_quant", torch::kCUDA,
           &persistent_masked_m_silu_mul_quant);

  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);

  // SM70/torch-2.10: old-ABI cuda_view in _C (stable ABI on torch 2.10 lacks a
  // from_blob deleter). Registered for both CUDA and ROCm.
  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
  ops.impl("get_cuda_view_from_cpu_tensor", torch::kCPU,
           &get_cuda_view_from_cpu_tensor);

  // Activation ops (quantized only — basic ops moved to _C_stable_libtorch)
  ops.def(
      "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()");
  ops.impl("silu_and_mul_quant", torch::kCUDA, &silu_and_mul_quant);

  // Horizontally-fused DeepseekV4-MLA: per-head RMSNorm + GPT-J RoPE for Q, and
  // GPT-J RoPE + UE8M0 FP8 quant + paged cache insert for KV, all in one
  // kernel launch. Registered in _C_stable_libtorch (incl. the FlashInfer V4
  // full-cache bf16/fp8 variants).

  // Quantization ops
#ifndef USE_ROCM

  // Note about marlin kernel 'workspace' arguments:
  // Technically these should be mutable since they are modified by the kernel.
  // But since they are set back to zero once the kernel is finished we can
  // hand wave and say that they have no net effect.
  //
  // The reason to mark 'workspace' as immutable is so that they don't interfere
  // with using ScalarType arguments in the ops. If they are marked as mutable,
  // pytorch throws an assert in
  // 'torch._higher_order_ops._register_effectful_op' that prevents these
  // kernels from being torch.compile'd.
  // See the following document for more info on custom types and ops that use
  // custom types:
  // https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA

  // Machete (Dense) Optimized Mixed Precision GEMM for Hopper.
  ops.def(
      "machete_supported_schedules("
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? maybe_group_scales_type,"
      "   ScalarType? maybe_group_zeros_type,"
      "   ScalarType? maybe_channel_scales_type,"
      "   ScalarType? maybe_token_scales_type,"
      "   ScalarType? maybe_out_type"
      ") -> str[]");
  ops.def(
      "machete_mm("
      "   Tensor A,"
      "   Tensor B,"
      "   int b_type,"
      "   ScalarType? out_type,"
      "   Tensor? group_scales,"
      "   Tensor? group_zeros,"
      "   int?    group_size,"
      "   Tensor? channel_scales,"
      "   Tensor? token_scales,"
      "   str?    schedule"
      ") -> Tensor");
  ops.def(
      "machete_prepack_B("
      "   Tensor B,"
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? group_scales_type"
      ") -> Tensor");
  // conditionally compiled so impl registration is in source file

  // Marlin Optimized Quantized GEMM (supports GPTQ, AWQ, FP8, NVFP4, MXFP4).
  ops.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none,Tensor b_scales, "
      "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, "
      "Tensor? "
      "g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // gptq_marlin repack from GPTQ.
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // awq_marlin repack from AWQ.
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // preprocess W-int4A-fp8 weight for marlin kernel
  ops.def(
      "marlin_int4_fp8_preprocess(Tensor qweight, "
      "Tensor? qzeros_or_none, bool inplace) -> Tensor");
  // conditionally compiled so impl registrations are in source file

#endif

#ifndef USE_ROCM
  // Expert-specialization mxfp8 blockscaled grouped quantization (SM100+).
  ops.def(
      "mxfp8_experts_quant("
      " Tensor input, Tensor problem_sizes, Tensor expert_offsets,"
      " Tensor blockscale_offsets, Tensor! quant_output, Tensor! scale_factor)"
      " -> ()");
  // conditionally compiled so impl registration is in source file

  // Expert-specialization mxfp8 blockscaled grouped GEMM (SM100+).
  ops.def(
      "cutlass_mxfp8_grouped_mm("
      " Tensor a, Tensor b, Tensor sfa, Tensor sfb, Tensor! out,"
      " Tensor problem_sizes, Tensor expert_offsets, Tensor blockscale_offsets)"
      " -> ()");
  // conditionally compiled so impl registration is in source file

#endif

#ifndef USE_ROCM
  // SM70 (V100) TurboMind GEMM kernels (impl in csrc/quantization/awq/*sm70*.cu).
  ops.def(
      "awq_sm70_prepare(Tensor _kernel, Tensor _scaling_factors, Tensor _zeros, "
      "int group_size, bool interleave_gated_silu) -> Tensor[]");
  ops.impl("awq_sm70_prepare", torch::kCUDA, &awq_sm70_prepare);

  ops.def("sm70_f16_prepare(Tensor _kernel) -> Tensor[]");
  ops.impl("sm70_f16_prepare", torch::kCUDA, &sm70_f16_prepare);

  ops.def(
      "awq_gemm_sm70(Tensor _in_feats, Tensor _kernel, Tensor "
      "_scaling_factors, int group_size, int k_ld, int q_ld) -> Tensor");
  ops.impl("awq_gemm_sm70", torch::kCUDA, &awq_gemm_sm70);

  ops.def("sm70_f16_gemm(Tensor _in_feats, Tensor _kernel) -> Tensor");
  ops.impl("sm70_f16_gemm", torch::kCUDA, &sm70_f16_gemm);

  ops.def(
      "awq_gemm_sm70_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "Tensor _scaling_factors, int group_size, int k_ld, int q_ld, "
      "bool gated_silu) -> ()");
  ops.impl("awq_gemm_sm70_out", torch::kCUDA, &awq_gemm_sm70_out);

  ops.def(
      "sm70_f16_gemm_out(Tensor(a!) out, Tensor _in_feats, Tensor _kernel, "
      "int k_ld, bool gated_silu) -> ()");
  ops.impl("sm70_f16_gemm_out", torch::kCUDA, &sm70_f16_gemm_out);

  ops.def(
      "w8a16_sm70a_prepare(Tensor weight_int8, Tensor scales, Tensor zeros, "
      "int group_size) -> Tensor[]");
  ops.impl("w8a16_sm70a_prepare", torch::kCUDA, &w8a16_sm70a_prepare);

  ops.def(
      "w8a16_sm70a_gemm_out(Tensor(a!) out, Tensor in_feats, Tensor tm_weight, "
      "Tensor tm_scales, int group_size, int k_ld, int q_ld, int n, "
      "bool gated_silu) -> ()");
  ops.impl("w8a16_sm70a_gemm_out", torch::kCUDA, &w8a16_sm70a_gemm_out);

  ops.def("sm70_gemm_import_cache(Tensor device_hint, str path) -> int");
  ops.impl("sm70_gemm_import_cache", torch::kCUDA, &sm70_gemm_import_cache);

  ops.def("sm70_gemm_export_cache(Tensor device_hint, str path) -> int");
  ops.impl("sm70_gemm_export_cache", torch::kCUDA, &sm70_gemm_export_cache);

  ops.def(
      "awq_moe_build_strided_ptrs(Tensor tm_weights, Tensor tm_scales, "
      "int k_ld, int q_ld, int num_experts) -> Tensor[]");
  ops.impl("awq_moe_build_strided_ptrs", torch::kCUDA,
           &awq_moe_build_strided_ptrs);

  ops.def(
      "awq_moe_single_token_compact_prepare("
      "Tensor topk_ids, "
      "Tensor src_w13_ptrs_w_rows, Tensor src_w13_ptrs_s_rows, "
      "Tensor src_w2_ptrs_w_rows, Tensor src_w2_ptrs_s_rows, "
      "Tensor(a!) dst_w13_ptrs_w_rows, Tensor(b!) dst_w13_ptrs_s_rows, "
      "Tensor(c!) dst_w2_ptrs_w_rows, Tensor(d!) dst_w2_ptrs_s_rows, "
      "Tensor(e!) inv_permuted_idx) -> ()");
  ops.impl("awq_moe_single_token_compact_prepare", torch::kCUDA,
           &awq_moe_single_token_compact_prepare);

  ops.def(
      "awq_moe_single_token_sm70_out("
      "Tensor(a!) out, Tensor x, Tensor topk_weights, Tensor topk_ids, "
      "Tensor src_w13_ptrs_w_rows, Tensor src_w13_ptrs_s_rows, "
      "Tensor src_w2_ptrs_w_rows, Tensor src_w2_ptrs_s_rows, "
      "Tensor(b!) compact_input, Tensor(c!) intermediate, "
      "Tensor(d!) sorted_output, "
      "Tensor(e!) dst_w13_ptrs_w_rows, Tensor(f!) dst_w13_ptrs_s_rows, "
      "Tensor(g!) dst_w2_ptrs_w_rows, Tensor(h!) dst_w2_ptrs_s_rows, "
      "Tensor(i!) expert_offsets, Tensor(j!) inv_permuted_idx, "
      "int w13_k, int w13_n, int w2_k, int w2_n, int group_size, "
      "int hidden_logical_size) -> ()");
  ops.impl("awq_moe_single_token_sm70_out", torch::kCUDA,
           &awq_moe_single_token_sm70_out);

  ops.def(
      "awq_moe_gemm_sm70(Tensor sorted_input, Tensor expert_offsets, "
      "Tensor strided_ptrs_w, Tensor strided_ptrs_s, "
      "int num_experts, int k, int n, int group_size) -> Tensor");
  ops.impl("awq_moe_gemm_sm70", torch::kCUDA, &awq_moe_gemm_sm70);

  ops.def(
      "awq_moe_gemm_sm70_out(Tensor(a!) out, Tensor sorted_input, "
      "Tensor expert_offsets, Tensor strided_ptrs_w, Tensor strided_ptrs_s, "
      "int num_experts, int k, int n, int group_size, bool gated_silu) -> ()");
  ops.impl("awq_moe_gemm_sm70_out", torch::kCUDA, &awq_moe_gemm_sm70_out);
#endif
}

#ifdef USE_ROCM
TORCH_LIBRARY_FRAGMENT(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Quick Reduce all-reduce kernels (ROCm-only; stays on legacy _C).
  custom_ar.def(
      "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
      "cast_bf2half) -> ()");
  custom_ar.impl("qr_all_reduce", torch::kCUDA, &qr_all_reduce);

  custom_ar.def("init_custom_qr", &init_custom_qr);
  custom_ar.def("qr_destroy", &qr_destroy);
  custom_ar.def("qr_get_handle", &qr_get_handle);

  custom_ar.def("qr_open_handles(int _fa, Tensor[](b!) handles) -> ()");
  custom_ar.impl("qr_open_handles", torch::kCPU, &qr_open_handles);

  custom_ar.def("qr_max_size", &qr_max_size);
}

// TODO: Remove this once ROCm upgrade to torch 2.11.
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // Cuda utils
  // Gets the specified device attribute.
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.impl("get_device_attribute", &get_device_attribute);

  // Gets the maximum shared memory per block device attribute.
  cuda_utils.def(
      "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  &get_max_shared_memory_per_block_device_attribute);
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
