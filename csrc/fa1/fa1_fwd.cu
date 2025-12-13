#include <torch/extension.h>
#include <ATen/ATen.h>
#include <algorithm>
#include <limits>
#include <tuple>

namespace {
using torch::Tensor;

Tensor apply_causal_mask(
    const Tensor& scores,
    int64_t row_start,
    int64_t col_start,
    int64_t row_end,
    int64_t col_end) {
  const auto device = scores.device();
  auto r = torch::arange(
      row_start,
      row_end,
      torch::TensorOptions().device(device).dtype(torch::kLong));
  auto c = torch::arange(
      col_start,
      col_end,
      torch::TensorOptions().device(device).dtype(torch::kLong));
  auto mask = c.view({1, -1}) > r.view({-1, 1});
  return scores.masked_fill(mask, -std::numeric_limits<float>::infinity());
}
} // namespace

std::tuple<Tensor, Tensor> fa1_forward(
    Tensor q,
    Tensor k,
    Tensor v,
    bool causal,
    double softmax_scale,
    int64_t br,
    int64_t bc) {
  at::NoGradGuard guard;

  TORCH_CHECK(
      q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
      "Expected q, k, v to be 3D (BH, N, D)");
  TORCH_CHECK(
      q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
          q.size(1) == k.size(1) && q.size(1) == v.size(1) &&
          q.size(2) == k.size(2) && q.size(2) == v.size(2),
      "q, k, v must share shape");

  const auto bh = q.size(0);
  const auto n = q.size(1);
  const auto d = q.size(2);
  const auto float_opts =
      q.options().dtype(torch::kFloat).memory_format(torch::MemoryFormat::Contiguous);

  auto o = torch::zeros({bh, n, d}, q.options());
  auto lse = torch::empty({bh, n}, float_opts);

  for (int64_t bh_idx = 0; bh_idx < bh; ++bh_idx) {
    auto q_bh = q.select(0, bh_idx);
    auto k_bh = k.select(0, bh_idx);
    auto v_bh = v.select(0, bh_idx);
    auto o_bh = o.select(0, bh_idx);
    auto lse_bh = lse.select(0, bh_idx);

    for (int64_t row_start = 0; row_start < n; row_start += br) {
      const int64_t row_end = std::min<int64_t>(row_start + br, n);
      const int64_t row_len = row_end - row_start;

      auto q_block = q_bh.narrow(0, row_start, row_len).to(torch::kFloat);
      auto m_i = torch::full({row_len}, -std::numeric_limits<float>::infinity(), float_opts);
      auto l_i = torch::zeros({row_len}, float_opts);
      auto o_block = torch::zeros({row_len, d}, float_opts);

      for (int64_t col_start = 0; col_start < n; col_start += bc) {
        if (causal && (col_start >= row_start + br)) {
          break;
        }
        const int64_t col_end = std::min<int64_t>(col_start + bc, n);
        const int64_t col_len = col_end - col_start;

        auto k_block = k_bh.narrow(0, col_start, col_len).to(torch::kFloat);
        auto v_block = v_bh.narrow(0, col_start, col_len).to(torch::kFloat);

        auto scores = torch::matmul(q_block, k_block.transpose(0, 1)) * softmax_scale;
        if (causal && (col_start <= row_start && row_start < col_end)) {
          scores = apply_causal_mask(scores, row_start, col_start, row_end, col_end);
        }

        auto rowmax = std::get<0>(scores.max(-1));
        auto m_new = torch::maximum(m_i, rowmax);
        auto p = (scores - m_new.unsqueeze(1)).exp();
        auto l_new = (m_i - m_new).exp() * l_i + p.sum(-1);
        o_block = (m_i - m_new).exp().unsqueeze(1) * o_block +
            torch::matmul(p, v_block);

        m_i = m_new;
        l_i = l_new;
      }

      auto out = (o_block / l_i.unsqueeze(1)).to(q.scalar_type());
      o_bh.narrow(0, row_start, row_len).copy_(out);
      lse_bh.narrow(0, row_start, row_len).copy_(m_i + l_i.log());
    }
  }

  return {o, lse};
}
