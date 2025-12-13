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

std::tuple<Tensor, Tensor> fa3_forward_inner(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    bool causal,
    double softmax_scale,
    int64_t br,
    int64_t bc) {
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
} // namespace

std::tuple<Tensor, Tensor> fa3_forward(
    Tensor q,
    Tensor k,
    Tensor v,
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
  TORCH_CHECK(
      q.sizes() == k.sizes() && q.sizes() == v.sizes(),
      "q, k, v must share shape");

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

  return fa3_forward_inner(q_use, k_use, v_use, causal, softmax_scale, br, bc);
}
