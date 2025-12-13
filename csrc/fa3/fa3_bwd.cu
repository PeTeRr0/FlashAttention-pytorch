#include <torch/extension.h>
#include <ATen/ATen.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace {
using torch::Tensor;
using torch::indexing::Slice;

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

Tensor hadamard_inplace(Tensor x) {
  int64_t h = 1;
  while (h < x.size(-1)) {
    auto a = x.index({Slice(), Slice(), Slice(0, c10::nullopt, 2 * h)});
    auto c = x.index({Slice(), Slice(), Slice(h, c10::nullopt, 2 * h)});
    auto a_plus_c = a + c;
    auto a_minus_c = a - c;
    x.index_put_({Slice(), Slice(), Slice(0, c10::nullopt, 2 * h)}, a_plus_c);
    x.index_put_({Slice(), Slice(), Slice(h, c10::nullopt, 2 * h)}, a_minus_c);
    h *= 2;
  }
  return x;
}

std::pair<Tensor, Tensor> incoherent_process(const Tensor& q, const Tensor& k) {
  const int64_t d = q.size(2);
  if ((d & (d - 1)) != 0) {
    return {q, k};
  }

  auto sign =
      torch::arange(d, torch::TensorOptions().device(q.device()).dtype(torch::kInt))
          .remainder(2)
          .mul(2)
          .sub(1)
          .to(q.scalar_type());

  auto q2 = q * sign.view({1, 1, d});
  auto k2 = k * sign.view({1, 1, d});
  q2 = hadamard_inplace(q2.to(torch::kFloat)).to(q.scalar_type());
  k2 = hadamard_inplace(k2.to(torch::kFloat)).to(k.scalar_type());

  const double scale = std::sqrt(static_cast<double>(d));
  q2 = q2 / scale;
  k2 = k2 / scale;
  return {q2, k2};
}

Tensor block_absmax_scale(const Tensor& x, int64_t block) {
  const int64_t n = x.size(1);
  const int64_t nb = (n + block - 1) / block;
  auto scales =
      torch::empty({x.size(0), nb}, x.options().dtype(torch::kFloat));

  for (int64_t i = 0; i < nb; ++i) {
    const int64_t s = i * block;
    const int64_t len = std::min<int64_t>(block, n - s);
    auto blk = x.narrow(1, s, len);
    auto m = torch::amax(
        blk.abs().to(torch::kFloat), std::vector<int64_t>({1, 2}));
    scales.select(1, i).copy_(m.clamp_min(1e-6));
  }
  return scales;
}

Tensor block_quant_dequant(const Tensor& x, const Tensor& scales, int64_t block) {
  auto xq = torch::empty_like(x, x.options().dtype(torch::kFloat16));
  const int64_t n = x.size(1);
  const int64_t nb = scales.size(1);

  for (int64_t i = 0; i < nb; ++i) {
    const int64_t s = i * block;
    const int64_t len = std::min<int64_t>(block, n - s);
    auto sc = scales.select(1, i).to(torch::kFloat16).view({scales.size(0), 1, 1});
    auto y = x.narrow(1, s, len).to(torch::kFloat16) / sc;
    y = torch::clamp(y, -1.0, 1.0);
    xq.narrow(1, s, len).copy_(y * sc);
  }
  return xq;
}
} // namespace

std::tuple<Tensor, Tensor, Tensor> fa3_backward(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor o,
    Tensor do_,
    Tensor lse,
    bool causal,
    double softmax_scale,
    int64_t br,
    int64_t bc,
    int64_t stages,
    bool fp8) {
  at::NoGradGuard guard;
  (void)stages;

  TORCH_CHECK(
      q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
      "Expected q, k, v to be 3D (BH, N, D)");

  const auto bh = q.size(0);
  const auto n = q.size(1);
  const auto d = q.size(2);
  const auto float_opts =
      q.options().dtype(torch::kFloat).memory_format(torch::MemoryFormat::Contiguous);

  Tensor q_use = q;
  Tensor k_use = k;
  Tensor v_use = v;

  if (fp8) {
    auto qk = incoherent_process(q_use, k_use);
    q_use = qk.first;
    k_use = qk.second;

    auto sq = block_absmax_scale(q_use, br);
    auto sk = block_absmax_scale(k_use, bc);
    auto sv = block_absmax_scale(v_use, bc);

    q_use = block_quant_dequant(q_use, sq, br);
    k_use = block_quant_dequant(k_use, sk, bc);
    v_use = block_quant_dequant(v_use, sv, bc);
  }

  auto dq = torch::zeros({bh, n, d}, float_opts);
  auto dk = torch::zeros({bh, n, d}, float_opts);
  auto dv = torch::zeros({bh, n, d}, float_opts);

  auto dvec = (do_.to(torch::kFloat) * o.to(torch::kFloat)).sum(-1);

  for (int64_t bh_idx = 0; bh_idx < bh; ++bh_idx) {
    auto q_bh = q_use.select(0, bh_idx);
    auto k_bh = k_use.select(0, bh_idx);
    auto v_bh = v_use.select(0, bh_idx);
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
