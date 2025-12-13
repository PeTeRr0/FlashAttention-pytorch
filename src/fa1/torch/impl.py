import torch

def _merge_bh(x):
    if x.dim() == 3:
        return x, None
    b, h, n, d = x.shape
    return x.reshape(b * h, n, d), (b, h)

def _split_bh(x, bh_shape):
    if bh_shape is None:
        return x
    b, h = bh_shape
    bh, n, d = x.shape
    return x.reshape(b, h, n, d)

def _split_bh_lse(lse, bh_shape):
    if bh_shape is None:
        return lse
    b, h = bh_shape
    bh, n = lse.shape
    return lse.reshape(b, h, n)

def _causal_block_skip(row_start, br, col_start):
    return col_start >= row_start + br

def _apply_causal_mask(scores, row_start, col_start):
    br = scores.shape[0]
    bc = scores.shape[1]
    r = torch.arange(row_start, row_start + br, device=scores.device)[:, None]
    c = torch.arange(col_start, col_start + bc, device=scores.device)[None, :]
    return scores.masked_fill(c > r, float("-inf"))

def fa1_forward_torch(q, k, v, causal, softmax_scale, br, bc):
    bh, n, d = q.shape
    o = torch.empty((bh, n, d), device=q.device, dtype=q.dtype)
    lse = torch.empty((bh, n), device=q.device, dtype=torch.float32)

    for bh_idx in range(bh):
        for row_start in range(0, n, br):
            row_end = min(row_start + br, n)
            qi = q[bh_idx, row_start:row_end, :].to(torch.float32)

            m_i = torch.full((row_end - row_start,), float("-inf"), device=q.device, dtype=torch.float32)
            l_i = torch.zeros((row_end - row_start,), device=q.device, dtype=torch.float32)
            acc = torch.zeros((row_end - row_start, d), device=q.device, dtype=torch.float32)

            for col_start in range(0, n, bc):
                if causal and _causal_block_skip(row_start, br, col_start):
                    break
                col_end = min(col_start + bc, n)

                kj = k[bh_idx, col_start:col_end, :].to(torch.float32)
                vj = v[bh_idx, col_start:col_end, :].to(torch.float32)

                scores = torch.matmul(qi, kj.transpose(-2, -1)) * softmax_scale

                if causal and (col_start <= row_start < col_start + bc):
                    scores = _apply_causal_mask(scores, row_start, col_start)

                rowmax = torch.amax(scores, dim=-1)
                m_new = torch.maximum(m_i, rowmax)

                p = torch.exp(scores - m_new[:, None])
                l_new = torch.exp(m_i - m_new) * l_i + torch.sum(p, dim=-1)

                acc = torch.exp(m_i - m_new)[:, None] * acc + torch.matmul(p, vj)

                m_i = m_new
                l_i = l_new

            out_block = (acc / l_i[:, None]).to(q.dtype)
            o[bh_idx, row_start:row_end, :] = out_block
            lse[bh_idx, row_start:row_end] = m_i + torch.log(l_i)

    return o, lse

def fa1_backward_torch(q, k, v, o, do, lse, causal, softmax_scale, br, bc):
    bh, n, d = q.shape

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    dvec = torch.sum(do.to(torch.float32) * o.to(torch.float32), dim=-1)

    for bh_idx in range(bh):
        for col_start in range(0, n, bc):
            col_end = min(col_start + bc, n)

            kj = k[bh_idx, col_start:col_end].to(torch.float32)
            vj = v[bh_idx, col_start:col_end].to(torch.float32)

            dk_j = torch.zeros((col_end - col_start, d), device=q.device, dtype=torch.float32)
            dv_j = torch.zeros((col_end - col_start, d), device=q.device, dtype=torch.float32)

            for row_start in range(0, n, br):
                if causal and _causal_block_skip(row_start, br, col_start):
                    continue
                row_end = min(row_start + br, n)

                qi = q[bh_idx, row_start:row_end].to(torch.float32)
                doi = do[bh_idx, row_start:row_end].to(torch.float32)
                lsei = lse[bh_idx, row_start:row_end].to(torch.float32)
                dveci = dvec[bh_idx, row_start:row_end].to(torch.float32)

                scores = torch.matmul(qi, kj.transpose(-2, -1)) * softmax_scale

                if causal and (col_start <= row_start < col_start + bc):
                    scores = _apply_causal_mask(scores, row_start, col_start)

                p = torch.exp(scores - lsei[:, None])

                dv_j = dv_j + p.transpose(0, 1) @ doi

                dp = doi @ vj.transpose(0, 1)
                ds = p * (dp - dveci[:, None])

                dq[bh_idx, row_start:row_end] = dq[bh_idx, row_start:row_end] + (ds @ kj) * softmax_scale
                dk_j = dk_j + (ds.transpose(0, 1) @ qi) * softmax_scale

            dk[bh_idx, col_start:col_end] = dk[bh_idx, col_start:col_end] + dk_j
            dv[bh_idx, col_start:col_end] = dv[bh_idx, col_start:col_end] + dv_j

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)

class _FA1TorchFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, softmax_scale, br, bc):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        o, lse = fa1_forward_torch(q, k, v, causal, softmax_scale, br, bc)

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = bool(causal)
        ctx.softmax_scale = float(softmax_scale)
        ctx.br = int(br)
        ctx.bc = int(bc)

        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()

        dq, dk, dv = fa1_backward_torch(q, k, v, o, do, lse, ctx.causal, ctx.softmax_scale, ctx.br, ctx.bc)
        return dq, dk, dv, None, None, None, None

def fa1_torch(q, k, v, causal, softmax_scale, spec):
    qb, bh_shape = _merge_bh(q)
    kb, _ = _merge_bh(k)
    vb, _ = _merge_bh(v)

    o, lse = _FA1TorchFn.apply(qb, kb, vb, causal, softmax_scale, spec.br, spec.bc)

    o = _split_bh(o, bh_shape)
    lse = _split_bh_lse(lse, bh_shape)

    return o, lse