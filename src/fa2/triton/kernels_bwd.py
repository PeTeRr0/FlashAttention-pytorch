import triton
import triton.language as tl

@triton.jit
def fa2_bwd_dk_dv_kernel(
    Q, K, V, O, 
    DO, LSE, D, DQ, DK, DV, 
    stride_qm, stride_qd, stride_km, stride_kd, 
    stride_vm, stride_vd, stride_om, stride_od, 
    stride_dom, stride_dod, stride_lm, stride_dm, 
    stride_dqm, stride_dqd, stride_dkm, stride_dkd, 
    stride_dvm, stride_dvd, n_ctx, 
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

    for pid_m in range(0, tl.cdiv(n_ctx, BR)):
        row_start = pid_m * BR
        if CAUSAL and (col_start >= row_start + BR):
            continue

        m_offsets = row_start + tl.arange(0, BR)

        q_ptr = Q + pid_bh * stride_qm * n_ctx + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
        o_ptr = O + pid_bh * stride_om * n_ctx + m_offsets[:, None] * stride_om + d_offsets[None, :] * stride_od
        do_ptr = DO + pid_bh * stride_dom * n_ctx + m_offsets[:, None] * stride_dom + d_offsets[None, :] * stride_dod
        
        q = tl.load(q_ptr, mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)
        o = tl.load(o_ptr, mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)
        do = tl.load(do_ptr, mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head), other=0.0).to(tl.float32)

        lse_ptr = LSE + pid_bh * stride_lm * n_ctx + m_offsets * stride_lm
        lse = tl.load(lse_ptr, mask=(m_offsets < n_ctx), other=0.0).to(tl.float32)
        d_ptrs = D + pid_bh * stride_dm * n_ctx + m_offsets * stride_dm
        dvec = tl.load(d_ptrs, mask=(m_offsets < n_ctx), other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k))

        if CAUSAL:
            r = m_offsets[:, None]
            c = n_offsets[None, :]
            scores = tl.where(c > r, float("-inf"), scores)

        p = tl.exp(scores - lse[:, None])

        dv += tl.dot(tl.trans(p), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dvec[:, None])

        dq_update = tl.dot(ds, k)
        dq_ptr = DQ + pid_bh * stride_dqm * n_ctx + m_offsets[:, None] * stride_dqm + d_offsets[None, :] * stride_dqd
        tl.atomic_add(dq_ptr, dq_update.to(tl.float16), mask=(m_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head))

        dk += tl.dot(tl.trans(ds), q)

    dk_ptr = DK + pid_bh * stride_dkm * n_ctx + n_offsets[:, None] * stride_dkm + d_offsets[None, :] * stride_dkd
    dv_ptr = DV + pid_bh * stride_dvm * n_ctx + n_offsets[:, None] * stride_dvm + d_offsets[None, :] * stride_dvd
    tl.store(dk_ptr, dk.to(tl.float16), mask=(n_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head))
    tl.store(dv_ptr, dv.to(tl.float16), mask=(n_offsets[:, None] < n_ctx) & (d_offsets[None, :] < d_head))