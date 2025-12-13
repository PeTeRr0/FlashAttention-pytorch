import triton
import triton.language as tl

@triton.jit
def fa1_bwd_d_kernel(
    O, DO, D,
    stride_om, stride_od, stride_dom, stride_dod, stride_dm,
    n_ctx, d_head: tl.constexpr, BR: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    row_start = pid_m * BR
    m_offsets = row_start + tl.arange(0, BR)
    d_offsets = tl.arange(0, d_head)

    o_ptr = O + pid_bh * stride_om * n_ctx + m_offsets[:, None] * stride_om + d_offsets[None, :] * stride_od
    do_ptr = DO + pid_bh * stride_dom * n_ctx + m_offsets[:, None] * stride_dom + d_offsets[None, :] * stride_dod

    o = tl.load(o_ptr, mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)

    dvec = tl.sum(o * do, axis=1)

    d_ptr = D + pid_bh * stride_dm * n_ctx + m_offsets * stride_dm
    tl.store(d_ptr, dvec, mask=(m_offsets < n_ctx))

@triton.jit
def fa1_bwd_dk_dv_kernel(
    Q, K, V, O,
    DO, LSE, D, DQ, DK, DV,
    stride_qm, stride_qd, stride_km, stride_kd,
    stride_vm, stride_vd, stride_om, stride_od,
    stride_dom, stride_dod, stride_lm, stride_dm,
    stride_dqm, stride_dqd, stride_dkm, stride_dkd,
    stride_dvm, stride_dvd, n_ctx, softmax_scale,
    d_head: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, CAUSAL: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    col_start = pid_m * BC
    n_offsets = col_start + tl.arange(0, BC)
    d_offsets = tl.arange(0, d_head)

    k_ptr = K + pid_bh * stride_km * n_ctx + n_offsets[:, None] * stride_km + d_offsets[None, :] * stride_kd
    v_ptr = V + pid_bh * stride_vm * n_ctx + n_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vd

    k = tl.load(k_ptr, mask=(n_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)
    v = tl.load(v_ptr, mask=(n_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)

    dk = tl.zeros((BC, d_head), dtype=tl.float32)
    dv = tl.zeros((BC, d_head), dtype=tl.float32)

    for qblk in range(0, tl.cdiv(n_ctx, BR)):
        row_start = qblk * BR
        if CAUSAL and (col_start >= row_start + BR):
            continue

        m_offsets = row_start + tl.arange(0, BR)

        q_ptr = Q + pid_bh * stride_qm * n_ctx + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
        do_ptr = DO + pid_bh * stride_dom * n_ctx + m_offsets[:, None] * stride_dom + d_offsets[None, :] * stride_dod

        q = tl.load(q_ptr, mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)
        do = tl.load(do_ptr, mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)

        lse_ptr = LSE + pid_bh * stride_lm * n_ctx + m_offsets * stride_lm
        lse = tl.load(lse_ptr, mask=(m_offsets < n_ctx), other=0.0).to(tl.float32)

        d_ptrs = D + pid_bh * stride_dm * n_ctx + m_offsets * stride_dm
        dvec = tl.load(d_ptrs, mask=(m_offsets < n_ctx), other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * softmax_scale

        if CAUSAL:
            r = m_offsets[:, None]
            c = n_offsets[None, :]
            scores = tl.where(c > r, float("-inf"), scores)

        p = tl.exp(scores - lse[:, None])

        dv += tl.dot(tl.trans(p), do)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dvec[:, None])

        dq_update = tl.dot(ds, k) * softmax_scale
        dq_ptr = DQ + pid_bh * stride_dqm * n_ctx + m_offsets[:, None] * stride_dqm + d_offsets[None, :] * stride_dqd
        tl.atomic_add(dq_ptr, dq_update, mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head))

        dk += tl.dot(tl.trans(ds), q) * softmax_scale

    dk_ptr = DK + pid_bh * stride_dkm * n_ctx + n_offsets[:, None] * stride_dkm + d_offsets[None, :] * stride_dkd
    dv_ptr = DV + pid_bh * stride_dvm * n_ctx + n_offsets[:, None] * stride_dvm + d_offsets[None, :] * stride_dvd

    tl.store(dk_ptr, dk, mask=(n_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head))
    tl.store(dv_ptr, dv, mask=(n_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head))
