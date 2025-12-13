// PyTorch extension entry point for FlashAttention lab kernels.
#include <torch/extension.h>
#include <tuple>

// Forward declarations defined in the algorithm-specific translation units.
std::tuple<torch::Tensor, torch::Tensor> fa1_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double softmax_scale,
    int64_t br,
    int64_t bc);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fa1_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor do_,
    torch::Tensor lse,
    bool causal,
    double softmax_scale,
    int64_t br,
    int64_t bc);

std::tuple<torch::Tensor, torch::Tensor> fa2_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double softmax_scale,
    int64_t br,
    int64_t bc);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fa2_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor do_,
    torch::Tensor lse,
    bool causal,
    double softmax_scale,
    int64_t br,
    int64_t bc);

std::tuple<torch::Tensor, torch::Tensor> fa3_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double softmax_scale,
    int64_t br,
    int64_t bc,
    int64_t stages,
    bool fp8);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fa3_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor do_,
    torch::Tensor lse,
    bool causal,
    double softmax_scale,
    int64_t br,
    int64_t bc,
    int64_t stages,
    bool fp8);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa1_forward", &fa1_forward, "FA1 forward");
  m.def("fa1_backward", &fa1_backward, "FA1 backward");

  // FlashAttention-2 bindings keep the simple forward/backward naming.
  m.def("forward", &fa2_forward, "FA2 forward");
  m.def("backward", &fa2_backward, "FA2 backward");

  m.def("fa3_forward", &fa3_forward, "FA3 forward");
  m.def("fa3_backward", &fa3_backward, "FA3 backward");
}
