#include <torch/extension.h>

// Forward declaration — implemented in flash_decode_paged.cu
at::Tensor flash_attention_decode_paged(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const float softmax_scale,
    const int partition_size,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_paged_fwd", &flash_attention_decode_paged,
        "Two-stage partitioned decode attention for SM70 (V100)",
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("out"),
        py::arg("block_table"),
        py::arg("seq_lens"),
        py::arg("tmp_out"),
        py::arg("max_logits"),
        py::arg("exp_sums"),
        py::arg("softmax_scale"),
        py::arg("partition_size"),
        py::arg("kv_cache_dtype"),
        py::arg("k_scale"),
        py::arg("v_scale"));
}
