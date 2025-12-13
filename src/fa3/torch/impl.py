import torch

def _merge_bh(x):
    if x.dim() == 3:
        return x
    b, h, n, d = x.shape
    return x.reshape(b * h, n, d)

def _causal_block_skip(row_start, br, col_start):
    return col_start >= row_start + br

def _apply_causal_mask(scores, row_start, col_start):
    br = scores.shape[0]
    bc = scores.shape[1]
    r = torch.arange(row_start, row_start + br, device=scores.device)[:, None]
    c = torch.arange(col_start, col_start + bc, device=scores.device)[None, :]
    mask = c > r
    return scores.masked_fill(mask, float("-inf"))

def _block_absmax_scale(x, block, dim=-1, eps=1e-6):
    n = x.shape[-2]
    d = x.shape[-1]
    nb = (n + block - 1) // block
    scales = torch.empty((x.shape[0], nb), device=x.device, dtype=torch.float32)
    for i in range(nb):
        s = i * block
        e = min(s + block, n)
        blk = x[:, s:e, :]
        m = torch.amax(torch.abs(blk).to(torch.float32), dim=(1, 2))
        scales[:, i] = m.clamp_min(eps)
    return scales

def _block_quant_dequant(x, scales, block):
    xq = torch.empty_like(x, dtype=torch.float16)
    n = x.shape[-2]
    nb = scales.shape[1]
    for i in range(nb):
        s = i * block
        e = min(s + block, n)
        sc = scales[:, i].to(torch.float16)[:, None, None]
        y = x[:, s:e, :].to(torch.float16) / sc
        y = torch.clamp(y, -1.0, 1.0)
        xq[:, s:e, :] = y * sc
    return xq

def _hadamard_inplace(x):
    b, n, d = x.shape
    h = 1
    y = x
    while h < d:
        a = y[..., 0::2*h]
        c = y[..., h::2*h]
        y[..., 0::2*h] = a + c
        y[..., h::2*h] = a - c
        h *= 2
    return y

def _incoherent_process(q, k, seed=0):
    b, n, d = q.shape
    if (d & (d - 1)) != 0:
        return q, k
    g = torch.Generator(device=q.device)
    g.manual_seed(int(seed))
    s = torch.randint(0, 2, (d,), device=q.device, generator=g, dtype=torch.int32)
    s = (s * 2 - 1).to(q.dtype)
    q2 = q * s[None, None, :]
    k2 = k * s[None, None, :]
    q2 = _hadamard_inplace(q2.to(torch.float32)).to(q.dtype)
    k2 = _hadamard_inplace(k2.to(torch.float32)).to(k.dtype)
    q2 = q2 / (d ** 0.5)
    k2 = k2 / (d ** 0.5)
    return q2, k2

def fa3_forward_torch(q, k, v, causal, softmax_scale, br, bc):
    bh, n, d = q.shape
    o = torch.empty((bh, n, d), device=q.device, dtype=q.dtype)
    lse = torch.empty((bh, n), device=q.device, dtype=torch.float32)

    for bh_idx in range(bh):
        for row_start in range(0, n, br):
            row_end = min(row_start + br, n)
            q_block = q[bh_idx, row_start:row_end, :].to(torch.float32)

            m_i = torch.full((row_end - row_start,), float("-inf"), device=q.device, dtype=torch.float32)
            l_i = torch.zeros((row_end - row_start,), device=q.device, dtype=torch.float32)
            o_block = torch.zeros((row_end - row_start, d), device=q.device, dtype=torch.float32)

            for col_start in range(0, n, bc):
                if causal and _causal_block_skip(row_start, br, col_start):
                    break
                col_end = min(col_start + bc, n)

                k_block = k[bh_idx, col_start:col_end, :].to(torch.float32)
                v_block = v[bh_idx, col_start:col_end, :].to(torch.float32)

                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * softmax_scale

                if causal and (col_start <= row_start < col_start + bc):
                    scores = _apply_causal_mask(scores, row_start, col_start)

                rowmax = torch.amax(scores, dim=-1)
                m_new = torch.maximum(m_i, rowmax)

                p = torch.exp(scores - m_new[:, None])
                l_new = torch.exp(m_i - m_new) * l_i + torch.sum(p, dim=-1)

                o_block = torch.exp(m_i - m_new)[:, None] * o_block + torch.matmul(p, v_block)

                m_i = m_new
                l_i = l_new

            out = (o_block / l_i[:, None]).to(q.dtype)
            o[bh_idx, row_start:row_end, :] = out
            lse[bh_idx, row_start:row_end] = m_i + torch.log(l_i)

    return o, lse

def fa3_torch(q, k, v, causal, softmax_scale, spec, fp8):
    qb = _merge_bh(q)
    kb = _merge_bh(k)
    vb = _merge_bh(v)

    if fp8:
        qb2, kb2 = _incoherent_process(qb, kb, seed=0)
        sq = _block_absmax_scale(qb2, spec.br)
        sk = _block_absmax_scale(kb2, spec.bc)
        sv = _block_absmax_scale(vb, spec.bc)
        qb2 = _block_quant_dequant(qb2, sq, spec.br)
        kb2 = _block_quant_dequant(kb2, sk, spec.bc)
        vb2 = _block_quant_dequant(vb, sv, spec.bc)
        return fa3_forward_torch(qb2, kb2, vb2, causal, softmax_scale, spec.br, spec.bc)

    return fa3_forward_torch(qb, kb, vb, causal, softmax_scale, spec.br, spec.bc)
