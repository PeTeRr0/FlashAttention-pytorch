#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fa1_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double softmax_scale, int64_t br, int64_t bc);
std::vector<torch::Tensor> fa1_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor do_, torch::Tensor lse, bool causal, double softmax_scale, int64_t br, int64_t bc);

std::vector<torch::Tensor> fa2_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double softmax_scale, int64_t br, int64_t bc);
std::vector<torch::Tensor> fa2_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor do_, torch::Tensor lse, bool causal, double softmax_scale, int64_t br, int64_t bc);

std::vector<torch::Tensor> fa3_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double softmax_scale, int64_t br, int64_t bc, int64_t stages, bool fp8);
std::vector<torch::Tensor> fa3_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor do_, torch::Tensor lse, bool causal, double softmax_scale, int64_t br, int64_t bc, int64_t stages, bool fp8);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa1_forward", &fa1_forward);
    m.def("fa1_backward", &fa1_backward);

    m.def("forward", &fa2_forward);
    m.def("backward", &fa2_backward);

    m.def("fa3_forward", &fa3_forward);
    m.def("fa3_backward", &fa3_backward);
}
