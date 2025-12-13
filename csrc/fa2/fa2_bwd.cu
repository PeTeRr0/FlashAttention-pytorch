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

std::tuple<Tensor, Tensor, Tensor> fa2_backward(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor o,
    Tensor do_,
    Tensor lse,
    bool causal,
    double softmax_scale,
    int64_t br,
    int64_t bc) {
  at::NoGradGuard guard;

  TORCH_CHECK(
      q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
      "Expected q, k, v to be 3D (BH, N, D)");

  const auto bh = q.size(0);
  const auto n = q.size(1);
  const auto d = q.size(2);
  const auto float_opts =
      q.options().dtype(torch::kFloat).memory_format(torch::MemoryFormat::Contiguous);

  auto dq = torch::zeros({bh, n, d}, float_opts);
  auto dk = torch::zeros({bh, n, d}, float_opts);
  auto dv = torch::zeros({bh, n, d}, float_opts);

  auto dvec = (do_.to(torch::kFloat) * o.to(torch::kFloat)).sum(-1);

  for (int64_t bh_idx = 0; bh_idx < bh; ++bh_idx) {
    auto q_bh = q.select(0, bh_idx);
    auto k_bh = k.select(0, bh_idx);
    auto v_bh = v.select(0, bh_idx);
    auto do_bh = do_.select(0, bh_idx);
    auto lse_bh = lse.select(0, bh_idx);
    auto dvec_bh = dvec.select(0, bh_idx);
    auto dq_bh = dq.select(0, bh_idx);
    auto dk_bh = dk.select(0, bh_idx);
    auto dv_bh = dv.select(0, bh_idx);

    for (int64_t row_start = 0; row_start < n; row_start += bc) {
      const int64_t row_end = std::min<int64_t>(row_start + bc, n);
      const int64_t row_len = row_end - row_start;

      auto kj = k_bh.narrow(0, row_start, row_len).to(torch::kFloat);
      auto vj = v_bh.narrow(0, row_start, row_len).to(torch::kFloat);
      auto dk_j = torch::zeros({row_len, d}, float_opts);
      auto dv_j = torch::zeros({row_len, d}, float_opts);

      for (int64_t col_start = 0; col_start < n; col_start += br) {
        if (causal && (col_start >= row_start + br)) {
          continue;
        }
        const int64_t col_end = std::min<int64_t>(col_start + br, n);
        const int64_t col_len = col_end - col_start;

        auto qi = q_bh.narrow(0, col_start, col_len).to(torch::kFloat);
        auto doi = do_bh.narrow(0, col_start, col_len).to(torch::kFloat);
        auto lsei = lse_bh.narrow(0, col_start, col_len).to(torch::kFloat);
        auto dveci = dvec_bh.narrow(0, col_start, col_len).to(torch::kFloat);

        auto scores = torch::matmul(qi, kj.transpose(0, 1)) * softmax_scale;
        if (causal && (row_start <= col_start && col_start < row_start + bc)) {
          scores = apply_causal_mask(scores, col_start, row_start, col_end, row_end);
        }

        auto p = (scores - lsei.unsqueeze(1)).exp();

        dv_j += torch::matmul(p.transpose(0, 1), doi);
        auto dp = torch::matmul(doi, vj.transpose(0, 1));
        auto ds = p * (dp - dveci.unsqueeze(1));

        dq_bh.narrow(0, col_start, col_len)
            .add_(torch::matmul(ds, kj) * softmax_scale);
        dk_j += torch::matmul(ds.transpose(0, 1), qi) * softmax_scale;
      }

      dk_bh.narrow(0, row_start, row_len).add_(dk_j);
      dv_bh.narrow(0, row_start, row_len).add_(dv_j);
    }
  }

  return {
      dq.to(q.scalar_type()),
      dk.to(k.scalar_type()),
      dv.to(v.scalar_type())};
}
