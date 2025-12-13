import torch
import triton
import triton.language as tl


@triton.jit
def _flash_bwd_kernel(
    Q, K, V, DO,
    DQ, DK, DV,
    M, N, H,
    stride_qb, stride_qh, stride_ql, stride_qd,         # strides for Q
    stride_kb, stride_kh, stride_kl, stride_kd,         # strides for K
    stride_vb, stride_vh, stride_vl, stride_vd,         # strides for V
    stride_dob, stride_doh, stride_dol, stride_dod,     # strides for DO
    stride_dqb, stride_dqh, stride_dql, stride_dqd,     # strides for DQ
    stride_dkb, stride_dkh, stride_dkl, stride_dkd,     # strides for DK
    stride_dvb, stride_dvh, stride_dvl, stride_dvd,     # strides for DV
    causal: tl.constexpr,
    dropout_p: tl.constexpr,
    softmax_scale: tl.constexpr,
    softcap: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_DMODEL)

    q_ptr = Q + b * stride_qb + h * stride_qh + m_start * stride_ql
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    do_ptr = DO + b * stride_dob + h * stride_doh + m_start * stride_dol
    dq_ptr = DQ + b * stride_dqb + h * stride_dqh + m_start * stride_dql
    dk_ptr = DK + b * stride_dkb + h * stride_dkh
    dv_ptr = DV + b * stride_dvb + h * stride_dvh

    q = tl.load(
        q_ptr + m_offsets[:, None] * stride_ql + d_offsets[None, :] * stride_qd,
        mask=(m_offsets[:, None] < M) & (d_offsets[None, :] < BLOCK_DMODEL),
        other=0.0,
    )
    do = tl.load(
        do_ptr + m_offsets[:, None] * stride_dol + d_offsets[None, :] * stride_dod,
        mask=(m_offsets[:, None] < M) & (d_offsets[None, :] < BLOCK_DMODEL),
        other=0.0,
    )

    dq_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    num_blocks = (N + BLOCK_N - 1) // BLOCK_N
    last_start = (num_blocks - 1) * BLOCK_N

    for block_idx in range(0, num_blocks):
        n_start = last_start - block_idx * BLOCK_N
        n_offsets = n_start + tl.arange(0, BLOCK_N)

        k_block = tl.load(
            k_ptr + n_offsets[:, None] * stride_kl + d_offsets[None, :] * stride_kd,
            mask=(n_offsets[:, None] < N) & (d_offsets[None, :] < BLOCK_DMODEL),
            other=0.0,
        )
        v_block = tl.load(
            v_ptr + n_offsets[:, None] * stride_vl + d_offsets[None, :] * stride_vd,
            mask=(n_offsets[:, None] < N) & (d_offsets[None, :] < BLOCK_DMODEL),
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
        if softcap > 0:
            qk = tl.minimum(qk, softcap)
            qk = tl.maximum(qk, -softcap)

        if causal:
            q_idx = m_offsets[:, None]
            k_idx = (n_start + tl.arange(0, BLOCK_N))[None, :]
            qk = tl.where(k_idx > q_idx, float("-inf"), qk)

        p = tl.softmax(qk, axis=1)
        if dropout_p > 0.0:
            keep = tl.rand(p.shape) >= dropout_p
            p = p * keep / (1.0 - dropout_p)

        dp = tl.dot(do, tl.trans(v_block))
        delta = tl.sum(dp * p, axis=1)
        dS = p * (dp - delta[:, None])

        dq_acc += tl.dot(dS, k_block)
        dk_block = tl.dot(tl.trans(dS), q)
        dv_block = tl.dot(tl.trans(p), do)

        tl.store(
            dk_ptr + n_offsets[:, None] * stride_dkl + d_offsets[None, :] * stride_dkd,
            dk_block,
            mask=(n_offsets[:, None] < N) & (d_offsets[None, :] < BLOCK_DMODEL),
        )
        tl.store(
            dv_ptr + n_offsets[:, None] * stride_dvl + d_offsets[None, :] * stride_dvd,
            dv_block,
            mask=(n_offsets[:, None] < N) & (d_offsets[None, :] < BLOCK_DMODEL),
        )

    tl.store(
        dq_ptr + m_offsets[:, None] * stride_dql + d_offsets[None, :] * stride_dqd,
        dq_acc,
        mask=(m_offsets[:, None] < M) & (d_offsets[None, :] < BLOCK_DMODEL),
    )

def flash_bwd(q, k, v, doutput, *, causal=False, dropout_p=0.0, softmax_scale=None, softcap=0.0):
    b, h, m, dmodel = q.shape
    n = k.shape[2]

    if softmax_scale is None:
        softmax_scale = dmodel ** -0.5

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if dmodel <= 64:
        block_m, block_n, block_dmodel = 64, 128, 64
    else:
        block_m, block_n, block_dmodel = 64, 64, min(128, dmodel)

    grid = (b * h, triton.cdiv(m, block_m))

    _flash_bwd_kernel[grid](
        q, k, v, doutput,
        dq, dk, dv,
        m, n, h,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        doutput.stride(0), doutput.stride(1), doutput.stride(2), doutput.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        causal,
        dropout_p,
        softmax_scale,
        softcap,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DMODEL=block_dmodel,
    )

    return dq, dk, dv