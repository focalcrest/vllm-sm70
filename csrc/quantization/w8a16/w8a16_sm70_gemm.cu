/*
 * SM70 W8A16 (weight-only INT8) GEMM using TurboMind s884h kernels.
 * Weight layout conversion + dequantized GEMM via HMMA_884 tensor cores.
 */

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace vllm {
namespace w8a16_sm70 {

namespace {

struct WorkspaceHolder {
  torch::Tensor barriers;
  torch::Tensor partials;
  torch::Tensor tensormaps;
  torch::Tensor flags;
  turbomind::gemm::Workspace workspace{};
};

struct GemmHolder {
  std::unique_ptr<turbomind::gemm::Gemm> gemm;
};

struct TuneKey {
  int device;
  int m;
  int n;
  int k;
  int group_size;

  bool operator==(const TuneKey& other) const {
    return device == other.device && m == other.m && n == other.n &&
           k == other.k && group_size == other.group_size;
  }
};

struct TuneKeyHash {
  std::size_t operator()(const TuneKey& key) const {
    std::size_t h = std::hash<int>()(key.device);
    h ^= std::hash<int>()(key.m) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>()(key.group_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct StreamWorkspaceKey {
  int device;
  cudaStream_t stream;

  bool operator==(const StreamWorkspaceKey& other) const {
    return device == other.device && stream == other.stream;
  }
};

struct StreamWorkspaceKeyHash {
  std::size_t operator()(const StreamWorkspaceKey& k) const {
    return std::hash<int>()(k.device) ^
           (std::hash<cudaStream_t>()(k.stream) << 1);
  }
};

std::mutex workspace_mutex;
std::mutex gemm_mutex;
std::mutex tune_mutex;
std::unordered_map<StreamWorkspaceKey, WorkspaceHolder, StreamWorkspaceKeyHash> workspace_cache;
std::unordered_map<int, GemmHolder> gemm_cache;
std::unordered_set<TuneKey, TuneKeyHash> tuned_shapes;
std::unordered_set<int> imported_cache_devices;

bool tune_small_shapes_enabled() {
  const char* raw = std::getenv("VLLM_SM70_W8A16_TUNE_SMALL_SHAPES");
  return raw == nullptr || std::atoi(raw) != 0;
}

int tune_max_m() {
  const char* raw = std::getenv("VLLM_SM70_W8A16_TUNE_MAX_M");
  return raw ? std::max(std::atoi(raw), 0) : 16;
}

bool is_stream_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const auto ec = cudaStreamIsCapturing(stream, &status);
  if (ec != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return status != cudaStreamCaptureStatusNone;
}

bool has_imported_cache(int device) {
  std::lock_guard<std::mutex> lock(tune_mutex);
  return imported_cache_devices.find(device) != imported_cache_devices.end();
}

turbomind::gemm::DispatchPolicy select_dispatch_policy(
    int device, int m, int n, int k, int group_size, cudaStream_t stream) {
  if (!tune_small_shapes_enabled() || m > tune_max_m()) {
    return turbomind::gemm::DispatchPolicy::kDefault;
  }

  TuneKey key{device, m, n, k, group_size};
  std::lock_guard<std::mutex> lock(tune_mutex);
  if (tuned_shapes.find(key) != tuned_shapes.end()) {
    return turbomind::gemm::DispatchPolicy::kReuse;
  }
  if (is_stream_capturing(stream)) {
    if (has_imported_cache(device)) {
      return turbomind::gemm::DispatchPolicy::kReuse;
    }
    return turbomind::gemm::DispatchPolicy::kDefault;
  }
  tuned_shapes.insert(key);
  return turbomind::gemm::DispatchPolicy::kMeasure;
}

WorkspaceHolder& get_workspace(int device, cudaStream_t stream) {
  StreamWorkspaceKey key{device, stream};

  {
    std::lock_guard<std::mutex> lock(workspace_mutex);
    auto it = workspace_cache.find(key);
    if (it != workspace_cache.end()) {
      return it->second;
    }
  }

  WorkspaceHolder holder;
  auto byte_opts = torch::TensorOptions()
                       .device(torch::Device(torch::kCUDA, device))
                       .dtype(torch::kUInt8);
  auto int_opts = torch::TensorOptions()
                      .device(torch::Device(torch::kCUDA, device))
                      .dtype(torch::kInt32);

  holder.barriers = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kBarriersSize}, byte_opts);
  holder.partials = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kPartialsSize}, byte_opts);
  holder.tensormaps = torch::empty({(long long)(8192 * 128)}, byte_opts);
  holder.flags = torch::zeros({1}, int_opts);

  holder.workspace.barriers = holder.barriers.data_ptr();
  holder.workspace.barriers_size = holder.barriers.numel();
  holder.workspace.partials = holder.partials.data_ptr();
  holder.workspace.partials_size = holder.partials.numel();
  holder.workspace.tensormaps = holder.tensormaps.data_ptr();
  holder.workspace.tensormaps_size = holder.tensormaps.numel();
  holder.workspace.flags = holder.flags.data_ptr<int>();

  std::lock_guard<std::mutex> lock(workspace_mutex);
  auto [insert_it, _] = workspace_cache.emplace(key, std::move(holder));
  return insert_it->second;
}

turbomind::gemm::Gemm& get_gemm(int device) {
  std::lock_guard<std::mutex> lock(gemm_mutex);
  auto it = gemm_cache.find(device);
  if (it != gemm_cache.end()) {
    return *it->second.gemm;
  }
  GemmHolder holder;
  holder.gemm = std::make_unique<turbomind::gemm::Gemm>();
  auto [insert_it, _] = gemm_cache.emplace(device, std::move(holder));
  return *insert_it->second.gemm;
}

}  // namespace

std::vector<torch::Tensor> w8a16_sm70_prepare(
    torch::Tensor weight_u8,
    torch::Tensor scales_f16,
    int64_t group_size) {
  TORCH_CHECK(weight_u8.is_cuda(), "w8a16_sm70_prepare: weight must be CUDA.");
  TORCH_CHECK(scales_f16.is_cuda(), "w8a16_sm70_prepare: scales must be CUDA.");
  TORCH_CHECK(weight_u8.scalar_type() == torch::kInt8,
              "w8a16_sm70_prepare: weight must be int8.");
  TORCH_CHECK(scales_f16.scalar_type() == torch::kFloat16,
              "w8a16_sm70_prepare: scales must be float16.");
  TORCH_CHECK(weight_u8.dim() == 2, "w8a16_sm70_prepare: weight must be 2D.");
  TORCH_CHECK(scales_f16.dim() == 2, "w8a16_sm70_prepare: scales must be 2D.");

  weight_u8 = weight_u8.contiguous();
  scales_f16 = scales_f16.contiguous();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight_u8));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t n = weight_u8.size(0);
  const int64_t k = weight_u8.size(1);
  const int64_t num_groups = scales_f16.size(0);

  TORCH_CHECK(scales_f16.size(1) == n,
              "w8a16_sm70_prepare: scales shape mismatch.");
  TORCH_CHECK(k % num_groups == 0,
              "w8a16_sm70_prepare: K must be divisible by num_groups.");

  if (group_size <= 0) {
    group_size = k / num_groups;
  }

  const bool grouped = (group_size != k);
  static const auto grouped_converters =
      turbomind::gemm::GetConverters(
          turbomind::kHalf, turbomind::kInt8, turbomind::kHalf, true, 70);
  static const auto ungrouped_converters =
      turbomind::gemm::GetConverters(
          turbomind::kHalf, turbomind::kInt8, turbomind::kHalf, false, 70);
  const auto& converters = grouped ? grouped_converters : ungrouped_converters;
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "w8a16_sm70_prepare: no compatible TurboMind converters.");

  // Weight layout conversion
  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kInt8,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout k_desc = w_desc;
  k_desc.pack = conv_w->pack;
  if (is_A_w) {
    k_desc = turbomind::gemm::transpose(k_desc);
  }

  // If converter expects row-major source but weight is [N,K] (row-major),
  // we may need to transpose for the source descriptor.
  auto weight_src = weight_u8;
  if (order_w == turbomind::gemm::kRowMajor) {
    // Source layout is row-major [N,K] which matches weight_u8 directly
  }

  // Compute output size from packed layout
  auto tm_weight = torch::empty_like(weight_u8);
  TORCH_CHECK(
      conv_w->Convert(weight_src.data_ptr(),
                      w_desc,
                      tm_weight.data_ptr(),
                      k_desc,
                      stream) == 0,
      "w8a16_sm70_prepare: weight conversion failed.");

  // Scale layout conversion
  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;
  const bool is_B_s = !is_A_s;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,
      order_s,
      static_cast<int>(n),
      static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (is_B_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout q_desc = s_desc;
  q_desc.pack = conv_s->pack;
  if (is_A_s) {
    q_desc = turbomind::gemm::transpose(q_desc);
  }

  // Convert FP16 scales to uint16 view for the converter
  auto scales_u16 = scales_f16.view(torch::kUInt16);
  auto tm_scales = torch::empty_like(scales_u16);
  TORCH_CHECK(
      conv_s->Convert(scales_u16.data_ptr(),
                      s_desc,
                      tm_scales.data_ptr(),
                      q_desc,
                      stream) == 0,
      "w8a16_sm70_prepare: scale conversion failed.");

  auto meta = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, k_desc.ld);
  meta.index_put_({1}, q_desc.ld);

  return {tm_weight, tm_scales, meta};
}

void w8a16_sm70_gemm_out(
    torch::Tensor out,
    torch::Tensor in_feats,
    torch::Tensor tm_weight,
    torch::Tensor tm_scales,
    int64_t group_size,
    int64_t w_ld,
    int64_t s_ld,
    bool gated_silu) {
  TORCH_CHECK(in_feats.is_cuda(), "w8a16_sm70_gemm: input must be CUDA.");
  TORCH_CHECK(tm_weight.is_cuda(), "w8a16_sm70_gemm: weight must be CUDA.");
  TORCH_CHECK(tm_scales.is_cuda(), "w8a16_sm70_gemm: scales must be CUDA.");
  TORCH_CHECK(out.is_cuda(), "w8a16_sm70_gemm: output must be CUDA.");
  TORCH_CHECK(in_feats.scalar_type() == torch::kFloat16,
              "w8a16_sm70_gemm: input must be float16.");
  TORCH_CHECK(tm_weight.scalar_type() == torch::kInt8,
              "w8a16_sm70_gemm: weight must be uint8.");
  TORCH_CHECK(tm_scales.scalar_type() == torch::kUInt16,
              "w8a16_sm70_gemm: scales must be uint16.");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16,
              "w8a16_sm70_gemm: output must be float16.");
  TORCH_CHECK(in_feats.dim() == 2, "w8a16_sm70_gemm: input must be 2D.");
  TORCH_CHECK(tm_weight.dim() == 2, "w8a16_sm70_gemm: weight must be 2D.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_feats));
  const int device = in_feats.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t m = in_feats.size(0);
  const int64_t k = in_feats.size(1);
  const int64_t n = tm_weight.size(0);

  TORCH_CHECK(tm_weight.size(1) == k,
              "w8a16_sm70_gemm: weight shape mismatch.");
  TORCH_CHECK(k % group_size == 0,
              "w8a16_sm70_gemm: input dim must be divisible by group size.");
  TORCH_CHECK(tm_scales.size(0) == k / group_size,
              "w8a16_sm70_gemm: scale groups mismatch.");
  TORCH_CHECK(tm_scales.size(1) == n,
              "w8a16_sm70_gemm: scale shape mismatch.");
  TORCH_CHECK(out.size(0) == m,
              "w8a16_sm70_gemm: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "w8a16_sm70_gemm: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "w8a16_sm70_gemm: gated_silu requires even output dim.");
    TORCH_CHECK(out.size(1) == n / 2,
                "w8a16_sm70_gemm: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "w8a16_sm70_gemm: output cols must match n.");
  }

  const bool grouped = (group_size != k);
  static const auto grouped_converters =
      turbomind::gemm::GetConverters(
          turbomind::kHalf, turbomind::kInt8, turbomind::kHalf, true, 70);
  static const auto ungrouped_converters =
      turbomind::gemm::GetConverters(
          turbomind::kHalf, turbomind::kInt8, turbomind::kHalf, false, 70);
  const auto& converters = grouped ? grouped_converters : ungrouped_converters;
  const auto* conv_w = converters[0];
  const auto* conv_s = converters[1];
  TORCH_CHECK(conv_w && conv_s,
              "w8a16_sm70_gemm: no compatible TurboMind converters.");

  // desc_A: FP16 input activations
  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(k),
  };
  turbomind::gemm::MatrixLayout desc_U{};

  // desc_B: uint8 weights
  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kInt8,
      order_w,
      static_cast<int>(n),
      static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (!is_A_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = static_cast<int>(w_ld);

  // desc_V: uint16 scales
  const auto order_s = conv_s->order;
  const bool is_A_s =
      turbomind::gemm::get_operand_tag(conv_s->pack) ==
      turbomind::gemm::OPERAND_U;

  const int64_t num_groups = k / group_size;

  turbomind::gemm::MatrixLayout s_desc{
      turbomind::kUint16,
      order_s,
      static_cast<int>(n),
      static_cast<int>(num_groups),
      static_cast<int>(n),
  };
  if (!is_A_s) {
    std::swap(s_desc.rows, s_desc.cols);
    s_desc.order = ~s_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_V = s_desc;
  desc_V.pack = conv_s->pack;
  if (is_A_s) {
    desc_V = turbomind::gemm::transpose(desc_V);
  }
  desc_V.ld = static_cast<int>(s_ld);

  // desc_D: FP16 output
  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };

  turbomind::gemm::Operation op{};
  op.dispatch = select_dispatch_policy(
      device, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      static_cast<int>(group_size), stream);
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kK, static_cast<int>(group_size)};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op,
                          1.f,
                          in_feats.data_ptr(),
                          desc_A,
                          nullptr,
                          desc_U,
                          tm_weight.data_ptr(),
                          desc_B,
                          tm_scales.data_ptr(),
                          desc_V,
                          0.f,
                          out.data_ptr(),
                          desc_D,
                          out.data_ptr(),
                          desc_D,
                          workspace_holder.workspace,
                          stream);
  TORCH_CHECK(ec == 0, "w8a16_sm70_gemm: TurboMind GEMM failed.");
}

}  // namespace w8a16_sm70
}  // namespace vllm

// C-level bindings for pybind
std::vector<torch::Tensor> w8a16_sm70_prepare(
    torch::Tensor weight_u8,
    torch::Tensor scales_f16,
    int64_t group_size) {
  return vllm::w8a16_sm70::w8a16_sm70_prepare(
      weight_u8, scales_f16, group_size);
}

void w8a16_sm70_gemm_out(
    torch::Tensor out,
    torch::Tensor in_feats,
    torch::Tensor tm_weight,
    torch::Tensor tm_scales,
    int64_t group_size,
    int64_t w_ld,
    int64_t s_ld,
    bool gated_silu) {
  vllm::w8a16_sm70::w8a16_sm70_gemm_out(
      out, in_feats, tm_weight, tm_scales, group_size, w_ld, s_ld, gated_silu);
}
