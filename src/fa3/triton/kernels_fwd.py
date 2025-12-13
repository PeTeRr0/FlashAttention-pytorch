import triton
import triton.language as tl

@triton.jit
def fa3_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qm, stride_qd, stride_km, stride_kd,
    stride_vm, stride_vd, stride_om, stride_od,
    stride_lm, n_ctx, softmax_scale,
    d_head: tl.constexpr, BR: tl.constexpr,
    BC: tl.constexpr, CAUSAL: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    row_start = pid_m * BR
    m_offsets = row_start + tl.arange(0, BR)
    d_offsets = tl.arange(0, d_head)

    q_ptr = Q + pid_bh * stride_qm * n_ctx + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q = tl.load(q_ptr, mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)

    m_i = tl.full((BR,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BR,), dtype=tl.float32)
    o = tl.zeros((BR, d_head), dtype=tl.float32)

    for j in range(0, tl.cdiv(n_ctx, BC)):
        col_start = j * BC
        if CAUSAL and (col_start >= row_start + BR):
            break

        n_offsets = col_start + tl.arange(0, BC)
        k_ptr = K + pid_bh * stride_km * n_ctx + n_offsets[:, None] * stride_km + d_offsets[None, :] * stride_kd
        v_ptr = V + pid_bh * stride_vm * n_ctx + n_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vd

        k = tl.load(k_ptr, mask=(n_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)
        v = tl.load(v_ptr, mask=(n_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * softmax_scale

        if CAUSAL:
            r = m_offsets[:, None]
            c = n_offsets[None, :]
            scores = tl.where(c > r, float("-inf"), scores)

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)
        o = tl.exp(m_i - m_new)[:, None] * o + tl.dot(p, v)

        m_i = m_new
        l_i = l_new

    o = o / l_i[:, None]
    lse = m_i + tl.log(l_i)

    o_ptr = O + pid_bh * stride_om * n_ctx + m_offsets[:, None] * stride_om + d_offsets[None, :] * stride_od
    tl.store(o_ptr, o.to(tl.float16), mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head))

    lse_ptr = LSE + pid_bh * stride_lm * n_ctx + m_offsets * stride_lm
    tl.store(lse_ptr, lse, mask=(m_offsets < n_ctx))
