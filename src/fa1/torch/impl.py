import torch

def _merge_bh(x):
    if x.dim() == 3:
        return x
    b, h, n, d = x.shape
    return x.reshape(b * h, n, d)

def _split_bh(x, b, h):
    if x.dim() == 3:
        bh, n, d = x.shape
        return x.reshape(b, h, n, d)
    return x

def _causal_block_skip(row_start, br, col_start):
    return col_start >= row_start + br

def _apply_causal_mask(scores, row_start, col_start):
    br = scores.shape[0]
    bc = scores.shape[1]
    r = torch.arange(row_start, row_start + br, device=scores.device)[:, None]
    c = torch.arange(col_start, col_start + bc, device=scores.device)[None, :]
    mask = c > r
    return scores.masked_fill(mask, float("-inf"))

def fa1_forward_torch(q, k, v, causal, softmax_scale, br, bc):
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

def fa1_backward_torch(q, k, v, o, do, lse, causal, softmax_scale, br, bc):
    bh, n, d = q.shape
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    dvec = torch.sum(do.to(torch.float32) * o.to(torch.float32), dim=-1)

    for bh_idx in range(bh):
        for row_start in range(0, n, bc):
            row_end = min(row_start + bc, n)

            kj = k[bh_idx, row_start:row_end].to(torch.float32)
            vj = v[bh_idx, row_start:row_end].to(torch.float32)

            dk_j = torch.zeros((row_end - row_start, d), device=q.device, dtype=torch.float32)
            dv_j = torch.zeros((row_end - row_start, d), device=q.device, dtype=torch.float32)

            for col_start in range(0, n, br):
                if causal and _causal_block_skip(col_start, br, row_start):
                    continue
                col_end = min(col_start + br, n)

                qi = q[bh_idx, col_start:col_end].to(torch.float32)
                doi = do[bh_idx, col_start:col_end].to(torch.float32)
                lsei = lse[bh_idx, col_start:col_end].to(torch.float32)
                dveci = dvec[bh_idx, col_start:col_end].to(torch.float32)

                scores = torch.matmul(qi, kj.transpose(-2, -1)) * softmax_scale

                if causal and (row_start <= col_start < row_start + bc):
                    scores = _apply_causal_mask(scores, col_start, row_start)

                p = torch.exp(scores - lsei[:, None])

                dv_j = dv_j + p.transpose(0, 1) @ doi
                dp = doi @ vj.transpose(0, 1)
                ds = p * (dp - dveci[:, None])

                dq[bh_idx, col_start:col_end] = dq[bh_idx, col_start:col_end] + (ds @ kj) * softmax_scale
                dk_j = dk_j + (ds.transpose(0, 1) @ qi) * softmax_scale

            dk[bh_idx, row_start:row_end] = dk[bh_idx, row_start:row_end] + dk_j
            dv[bh_idx, row_start:row_end] = dv[bh_idx, row_start:row_end] + dv_j

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)

def fa1_torch(q, k, v, causal, softmax_scale, spec):
    qb = _merge_bh(q)
    kb = _merge_bh(k)
    vb = _merge_bh(v)
    o, lse = fa1_forward_torch(qb, kb, vb, causal, softmax_scale, spec.br, spec.bc)
    return o, lse
