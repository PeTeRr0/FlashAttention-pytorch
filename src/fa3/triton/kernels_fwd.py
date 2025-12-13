import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd_kernel(
    Q, K, V, O,
    M, N, H,
    stride_qb, stride_qh, stride_ql, stride_qd,  # strides for Q
    stride_kb, stride_kh, stride_kl, stride_kd,  # strides for K
    stride_vb, stride_vh, stride_vl, stride_vd,  # strides for V
    stride_ob, stride_oh, stride_ol, stride_od,  # strides for O
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
    o_ptr = O + b * stride_ob + h * stride_oh + m_start * stride_ol

    q = tl.load(
        q_ptr + m_offsets[:, None] * stride_ql + d_offsets[None, :] * stride_qd,
        mask=(m_offsets[:, None] < M) & (d_offsets[None, :] < BLOCK_DMODEL),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    k_next = tl.load(
        k_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_kl + d_offsets[None, :] * stride_kd,
        mask=(tl.arange(0, BLOCK_N)[:, None] < N) & (d_offsets[None, :] < BLOCK_DMODEL),
        other=0.0,
    )
    v_next = tl.load(
        v_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_vl + d_offsets[None, :] * stride_vd,
        mask=(tl.arange(0, BLOCK_N)[:, None] < N) & (d_offsets[None, :] < BLOCK_DMODEL),
        other=0.0,
    )

    for n_start in range(0, N, BLOCK_N):
        k_block = k_next
        v_block = v_next

        n_next = n_start + BLOCK_N
        if n_next < N:
            n_offsets_next = n_next + tl.arange(0, BLOCK_N)
            k_next = tl.load(
                k_ptr + n_offsets_next[:, None] * stride_kl + d_offsets[None, :] * stride_kd,
                mask=(n_offsets_next[:, None] < N) & (d_offsets[None, :] < BLOCK_DMODEL),
                other=0.0,
            )
            v_next = tl.load(
                v_ptr + n_offsets_next[:, None] * stride_vl + d_offsets[None, :] * stride_vd,
                mask=(n_offsets_next[:, None] < N) & (d_offsets[None, :] < BLOCK_DMODEL),
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

        block_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, block_max)
        exp_m = tl.exp(m_i - m_new)
        exp_qk = tl.exp(qk - m_new[:, None])
        l_new = exp_m * l_i + tl.sum(exp_qk, axis=1)

        p = exp_qk / l_new[:, None]
        if dropout_p > 0.0:
            keep = tl.rand(p.shape) >= dropout_p
            p = p * keep / (1.0 - dropout_p)

        acc = (exp_m * l_i / l_new)[:, None] * acc + tl.dot(p, v_block)

        m_i = m_new
        l_i = l_new

    tl.store(
        o_ptr + m_offsets[:, None] * stride_ol + d_offsets[None, :] * stride_od,
        acc,
        mask=(m_offsets[:, None] < M) & (d_offsets[None, :] < BLOCK_DMODEL),
    )


def flash_fwd(q, k, v, *, causal=False, dropout_p=0.0, softmax_scale=None, softcap=0.0):
    b, h, q_len, d_k = q.shape
    k_len = k.shape[2]

    if softmax_scale is None:
        softmax_scale = d_k ** -0.5

    # Hopper-optimized configuration favors smaller BLOCK_M to increase parallelism.
    if d_k <= 64:
        block_m, block_n, block_dmodel = 64, 128, 64
    else:
        block_m, block_n, block_dmodel = 64, 64, min(128, d_k)

    output = torch.empty_like(q)
    aux = torch.empty((0,), device=q.device)

    grid = (b * h, triton.cdiv(q_len, block_m))

    _flash_fwd_kernel[grid](
        q, k, v, output,
        q_len, k_len, h,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        causal,
        dropout_p,
        softmax_scale,
        softcap,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DMODEL=block_dmodel,
    )
    return output, aux