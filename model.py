#@title FlashAttention GPT-3

import torch
from torch import nn
import math
import time, gc
import random
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import tiktoken
from collections import Counter
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch import amp
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads, dropout, use_fused_qkv=True, block_size=128):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads
    self.block_size = block_size

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # fused QKV for efficient self-attention
    # projects to 3 * d_model and we split into Q, K, V for the common self-attn path
    self.use_fused_qkv = use_fused_qkv
    self.w_qkv = nn.Linear(d_model, 3 * d_model) if use_fused_qkv else None
    # keep separate projection linears available for cross-attention or clarity
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)

    # dropout applied to attention probabilities
    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v, tau=1.0, mask=None, block_sparse_mask=None):
    """
    for standard self-attention q,k,v inputs are the same tensor.
    We'll project q,k,v from their respective inputs but using the same fused layer
    q: (b_q, q_len, d_model)
    k: (b_k, kv_len, d_model)
    v: (b_k, kv_len, d_model)
    """

    b_q, q_len, _ = q.size()
    b_k, kv_len, _ = k.size()

    assert b_q == b_k, "Batch sizes for q and k must match"

    # --- projections ---
    # Fast path for self-attention when q,k,v are the same tensor and fused QKV is enabled
    if self.use_fused_qkv and (q is k is v):
        # self-attention common fast path: one linear -> split into Q,K,V
        qkv = self.w_qkv(q)                       # (B, S_q, 3*d_model)
        q_proj, k_proj, v_proj = qkv.chunk(3, dim=-1)
    else:
        # cross-attention or fused disabled: compute projections separately
        if self.use_fused_qkv:
          qkv = self.w_qkv(q)                     # (B, S_q, 3*d_model)
          q_proj = qkv[..., :self.d_model]        # (B, S_q, d_model)
        else:
          q_proj = self.w_q(q)                    # (B, S_q, d_model)
        k_proj = self.w_k(k)                      # (B, S_k, d_model)
        v_proj = self.w_v(v)                      # (B, S_k, d_model)

    # reshape to (B, H, S, d_k) for multi-head matmuls
    q = q_proj.view(b_q, q_len, self.num_heads, self.d_k).transpose(1, 2)
    k = k_proj.view(b_k, kv_len, self.num_heads, self.d_k).transpose(1, 2)
    v = v_proj.view(b_k, kv_len, self.num_heads, self.d_k).transpose(1, 2)

    # Block-sparse FlashAttention or standard attention
    if block_sparse_mask is not None:
        out = self._block_sparse_flash_attention(q, k, v, tau, mask, block_sparse_mask)
    else:
        # scaled dot-product attention (dense)
        scores = tau * torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: positions with 0 are set to -inf so softmax makes them zero
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) # (B, H, S_q, d_k)

    # combine heads and project output
    out = out.transpose(1, 2).reshape(b_q, q_len, self.d_model)
    out = self.w_o(out)
    return out

  def _block_sparse_flash_attention(self, q, k, v, tau=1.0, mask=None, block_sparse_mask=None):
    """Block-Sparse FlashAttention (Algorithm 5)"""
    b, num_heads, q_len, d_k = q.shape
    kv_len = k.shape[2]

    # Block sizes (columns/rows processed per iteration)
    Bc = min(self.block_size, kv_len)
    Br = min(self.block_size, q_len)

    # Initialize output accumulator O and softmax statistics l (denominator) and m (max logit)
    O = torch.zeros_like(q)
    l = torch.zeros(b, num_heads, q_len, 1, device=q.device)
    m = torch.full((b, num_heads, q_len, 1), float('-inf'), device=q.device)

    Tc = math.ceil(kv_len / Bc)
    Tr = math.ceil(q_len / Br)

    # Outer loop over K, V blocks
    for j in range(Tc):
        j_start = j * Bc
        j_end = min((j + 1) * Bc, kv_len)

        # Copy block-of-K and block-of-V
        # Load Kj, Vj from HBM to SRAM
        Kj = k[:, :, j_start:j_end, :]
        Vj = v[:, :, j_start:j_end, :]

        # Inner loop over Q blocks
        for i in range(Tr):
            # Skip zero blocks in block-sparse mask to save compute (Algorithm 5, line 8)
            if block_sparse_mask[i, j] == 0:
                continue

            i_start = i * Br
            i_end = min((i + 1) * Br, q_len)

            # Load Qi, Oi, li, mi
            Qi = q[:, :, i_start:i_end, :]
            Oi = O[:, :, i_start:i_end, :]
            li = l[:, :, i_start:i_end, :]
            mi = m[:, :, i_start:i_end, :]

            # Compute scaled dot-product Sij (Qi @ Kj^T)
            Sij = tau * torch.matmul(Qi, Kj.transpose(-2, -1)) / math.sqrt(d_k)

            # Apply mask
            if mask is not None:
                mask_block = mask[:, :, i_start:i_end, j_start:j_end]
                Sij = Sij.masked_fill(mask_block == 0, float('-inf'))

            # Compute numerically-stable exponentials relative to block max
            mij_tilde = torch.max(Sij, dim=-1, keepdim=True)[0]
            Pij_tilde = torch.exp(Sij - mij_tilde)  # positive values (no dropout applied yet)

            # --- sample dropout mask and rescale (match ForwardPass behavior) ---
            p = float(self.dropout.p) if self.training else 0.0
            if p > 0.0 and self.training:
                rnd = torch.rand_like(Pij_tilde)
                mask_ij = (rnd > p)
                Pij_dropped = Pij_tilde * mask_ij.to(Pij_tilde.dtype) / (1.0 - p)
            else:
                mask_ij = None
                Pij_dropped = Pij_tilde

            # Use the dropped (and rescaled) values when computing sums/outputs
            lij_tilde = torch.sum(Pij_dropped, dim=-1, keepdim=True)

            # Update mi_new, li_new in a numerically stable way
            mi_new = torch.max(mi, mij_tilde)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_tilde - mi_new) * lij_tilde

            # Update Oi using dropped Pij (consistent with forward)
            Oi_new = (Oi * torch.exp(mi - mi_new) * li +
                      torch.matmul(Pij_dropped, Vj) * torch.exp(mij_tilde - mi_new)) / li_new

            # Write back updated accumulators and output block
            O[:, :, i_start:i_end, :] = Oi_new
            l[:, :, i_start:i_end, :] = li_new
            m[:, :, i_start:i_end, :] = mi_new

    return O

def look_ahead_mask_(q_len, k_len=None, device=None):
    """
    Improved causal mask:
      - supports q_len != k_len (useful when using cached past key/values)
      - returns a boolean mask of shape (1, 1, q_len, k_len) where True = allowed, False = masked
    """
    if k_len is None:
        k_len = q_len
    device = device if device is not None else torch.device('cpu')

    q_idx = torch.arange(q_len, device=device).unsqueeze(1)   # (q_len, 1)
    k_idx = torch.arange(k_len, device=device).unsqueeze(0)   # (1, k_len)
    offset = k_len - q_len
    mask = (k_idx <= (q_idx + offset))                        # (q_len, k_len)
    return mask.unsqueeze(0).unsqueeze(0)                     # (1, 1, q_len, k_len)

class Decoder(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout, use_fused_qkv=True, block_size=128):
    super().__init__()

    self.attn = MultiHeadAttention(d_model, num_heads, dropout, use_fused_qkv, block_size) #Masked MHA
    self.dropout1 = nn.Dropout(dropout)
    self.layer_norm1 = nn.LayerNorm(d_model)

    self.ffn = FeedForward(d_model, num_heads, dropout, block_size)
    self.dropout2 = nn.Dropout(dropout)
    self.layer_norm2 = nn.LayerNorm(d_model)

  def forward(self, x, look_ahead_mask_=None, tau=1.0):
    attention_out = self.attn(x, x, x, tau, look_ahead_mask_)
    x = x + self.dropout1(attention_out)
    x = self.layer_norm1(x)

    ffn_out = self.ffn(x)
    x = x + self.dropout2(ffn_out)
    x = self.layer_norm2(x)

    return x

class DecoderStack(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, use_fused_qkv, block_size=128):
        super().__init__()
        self.layers = nn.ModuleList([
            Decoder(d_model, num_heads, d_ff, dropout, use_fused_qkv, block_size)
            for _ in range(num_layers)
        ])

    def forward(self, x, look_ahead_mask_=None, tau=1.0):
        for layer in self.layers:
            x = layer(x, look_ahead_mask_, tau)
        return x

class ForwardPass(nn.Module):
    def __init__(self, d_model, num_heads, dropout, block_size=128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, q, k, v, tau=1.0, mask=None, block_sparse_mask=None):
        """
        q, k, v: (B, N, d_model)
        mask: (B, 1, N, N)
        block_sparse_mask: (num_blocks_Q, num_blocks_K)
        """
        B, N, _ = q.shape
        H = self.num_heads
        d_k = self.d_k

        # reshape for multi-head
        q = q.view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        k = k.view(B, N, H, d_k).transpose(1, 2)
        v = v.view(B, N, H, d_k).transpose(1, 2)

        # block sizes
        Br = min(self.block_size or N, N)
        Bc = min(self.block_size or N, N)

        Tr = math.ceil(N / Br)
        Tc = math.ceil(N / Bc)

        # initialize outputs and softmax stats
        O = torch.zeros_like(q)
        l = torch.zeros(B, H, N, 1, device=q.device)
        m = torch.full((B, H, N, 1), float('-inf'), device=q.device)

        masks = []

        for j in range(Tc):
            j_start = j * Bc
            j_end = min((j + 1) * Bc, N)
            Kj = k[:, :, j_start:j_end, :]
            Vj = v[:, :, j_start:j_end, :]

            for i in range(Tr):
                if block_sparse_mask is not None and block_sparse_mask[i, j] == 0:
                    continue

                i_start = i * Br
                i_end = min((i + 1) * Br, N)

                Qi = q[:, :, i_start:i_end, :]
                Oi = O[:, :, i_start:i_end, :]
                li = l[:, :, i_start:i_end, :]
                mi = m[:, :, i_start:i_end, :]

                # scaled dot-product
                Sij = tau * torch.matmul(Qi, Kj.transpose(-2, -1)) / math.sqrt(d_k)

                if mask is not None:
                    Sij = Sij.masked_fill(mask[:, :, i_start:i_end, j_start:j_end] == 0, float('-inf'))

                # numerically stable exponent
                mij_tilde = torch.max(Sij, dim=-1, keepdim=True)[0]
                Pij_tilde = torch.exp(Sij - mij_tilde)  # positive values (no dropout applied yet)

                # --- explicit block dropout mask (sample once and save) ---
                p = float(self.dropout.p) if self.training else 0.0

                if p > 0.0 and self.training:
                    # sample binary mask with same shape as Pij_tilde, then rescale to keep expectation
                    rnd = torch.rand_like(Pij_tilde)
                    mask_ij = (rnd > p) # bool tensor
                    Pij_dropped = Pij_tilde * mask_ij.to(Pij_tilde.dtype) / (1.0 - p)
                else:
                    mask_ij = None
                    Pij_dropped = Pij_tilde

                # save mask entry (keeps one-to-one ordering with block traversal)
                # masks is a list declared earlier: masks = []
                masks.append(mask_ij)

                # IMPORTANT: use dropped version to compute sums (consistency)
                lij_tilde = torch.sum(Pij_dropped, dim=-1, keepdim=True)

                mi_new = torch.max(mi, mij_tilde)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_tilde - mi_new) * lij_tilde

                Oi_new = (Oi * torch.exp(mi - mi_new) * li +
                          torch.matmul(Pij_dropped, Vj) * torch.exp(mij_tilde - mi_new)) / li_new

                # write back
                O[:, :, i_start:i_end, :] = Oi_new
                l[:, :, i_start:i_end, :] = li_new
                m[:, :, i_start:i_end, :] = mi_new

        return O.transpose(1, 2).reshape(B, N, self.d_model), l, m, masks

class BackwardPass(nn.Module):
    def __init__(self, d_model, num_heads, dropout, block_size=128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, q, k, v, dO, tau=1.0, mask=None, block_sparse_mask=None, masks=None):
        """
        q, k, v: (B, N, d_model)
        dO: gradient of output (B, N, d_model)
        mask: (B, 1, N, N)
        block_sparse_mask: (num_blocks_Q, num_blocks_K)
        Returns:
            dQ, dK, dV: gradients w.r.t Q, K, V
        """
        B, N, _ = q.shape
        H = self.num_heads
        d_k = self.d_k

        # reshape for multi-head
        q = q.view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        k = k.view(B, N, H, d_k).transpose(1, 2)
        v = v.view(B, N, H, d_k).transpose(1, 2)
        dO = dO.view(B, N, H, d_k).transpose(1, 2)

        Br = min(self.block_size or N, N)
        Bc = min(self.block_size or N, N)
        Tr = math.ceil(N / Br)
        Tc = math.ceil(N / Bc)

        # initialize gradients
        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)

        # softmax statistics placeholders
        l = torch.zeros(B, H, N, 1, device=q.device)
        m = torch.full((B, H, N, 1), float('-inf'), device=q.device)

        mask_idx = 0

        for j in range(Tc):
          j_start = j * Bc
          j_end = min((j + 1) * Bc, N)
          Kj = k[:, :, j_start:j_end, :]
          Vj = v[:, :, j_start:j_end, :]

          # accumulate gradients for this Kj,Vj across i's, then write once after the i-loop
          dKj = torch.zeros_like(Kj)
          dVj = torch.zeros_like(Vj)

          for i in range(Tr):
              if block_sparse_mask is not None and block_sparse_mask[i, j] == 0:
                  continue

              i_start = i * Br
              i_end = min((i + 1) * Br, N)

              Qi = q[:, :, i_start:i_end, :]
              dOi = dO[:, :, i_start:i_end, :]
              li = l[:, :, i_start:i_end, :]
              mi = m[:, :, i_start:i_end, :]

              # scaled dot-product (recompute)
              Sij = tau * torch.matmul(Qi, Kj.transpose(-2, -1)) / math.sqrt(d_k)
              if mask is not None:
                  Sij = Sij.masked_fill(mask[:, :, i_start:i_end, j_start:j_end] == 0, float('-inf'))

              mij_tilde = torch.max(Sij, dim=-1, keepdim=True)[0]
              Pij = torch.exp(Sij - mij_tilde)
              # retrieve the same dropout mask used in forward
              mask_ij = None
              if masks is not None:
                  mask_ij = masks[mask_idx]  # masks defined/passed in outer scope
                  mask_idx += 1

              # if mask present, apply same binary mask and rescale
              p = float(self.dropout.p) if self.training else 0.0

              if mask_ij is not None:
                  Pij = Pij * mask_ij / (1.0 - p)

              # normalized attention (A)
              lij = torch.sum(Pij, dim=-1, keepdim=True)
              Pij_norm = Pij / (lij + 1e-12)  # avoid divide-by-zero

              # backward: dA = dOi @ Vj^T
              dA = torch.matmul(dOi, Vj.transpose(-2, -1))

              # softmax jacobian: dS = A * (dA - sum(dA * A, dim=-1, keepdim=True))
              tmp = torch.sum(dA * Pij_norm, dim=-1, keepdim=True)
              dS = Pij_norm * (dA - tmp)

              # accumulate gradient w.r.t Vj across i
              dVj += torch.matmul(Pij_norm.transpose(-2, -1), dOi)

              # include tau scaling consistently (forward used tau / sqrt(d_k))
              scale = tau / math.sqrt(d_k)
              dQi = torch.matmul(dS, Kj) * scale
              dKj += torch.matmul(dS.transpose(-2, -1), Qi) * scale

              # accumulate gradient for Q immediately (per-i block)
              dQ[:, :, i_start:i_end, :] += dQi

          # after finishing all i for this j, write the accumulated dKj,dVj once
          dK[:, :, j_start:j_end, :] += dKj
          dV[:, :, j_start:j_end, :] += dVj


        # reshape back to (B, N, d_model)
        dQ = dQ.transpose(1, 2).reshape(B, N, self.d_model)
        dK = dK.transpose(1, 2).reshape(B, N, self.d_model)
        dV = dV.transpose(1, 2).reshape(B, N, self.d_model)

        return dQ, dK, dV

class FeedForward(nn.Module):
    """
    FeedForward that fuses projections + block-wise FlashAttention forward/backward
    using the provided ForwardPass and BackwardPass modules.

    Usage: replace the old FeedForward class with this. It expects self.w_q, w_k, w_v, w_o
    to be nn.Linear modules (bias may be None).
    """
    def __init__(self, d_model, num_heads, dropout, block_size=128):
        super().__init__()
        # Projection & output layers
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

        # low-level optimized forward/backward implementations
        self.forward_pass = ForwardPass(d_model, num_heads, dropout, block_size)
        self.backward_pass = BackwardPass(d_model, num_heads, dropout, block_size)

    def forward(self, x, mask=None, block_sparse_mask=None, tau=1.0):
        """
        x: (B, N, d_model)
        mask: (B, 1, N, N) or None
        block_sparse_mask: (num_blocks_Q, num_blocks_K) or None

        This uses a custom autograd.Function to ensure the backward uses the
        optimized BackwardPass implementation and computes weight gradients
        for the projection / output linears manually.
        """
        # local alias to make call-sites shorter
        return _FlashAttnFn.apply(
            x,
            self.w_q.weight, self.w_q.bias,
            self.w_k.weight, self.w_k.bias,
            self.w_v.weight, self.w_v.bias,
            self.w_o.weight, self.w_o.bias,
            self.forward_pass, self.backward_pass,
            tau, mask, block_sparse_mask
        )


class _FlashAttnFn(torch.autograd.Function):
    """
    Custom autograd Function:
      forward:  x -> (q,k,v) via linear weights -> ForwardPass -> out @ W_o^T + b_o
      backward: compute d_out -> d_output_attn -> use BackwardPass to get dQ,dK,dV
                then compute gradients for weights/biases and input x analytically.
    Note: mask and block_sparse_mask are treated as non-differentiable inputs (grad None).
    """

    @staticmethod
    def forward(ctx, x,
                wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b,
                forward_pass, backward_pass,
                tau, mask, block_sparse_mask):
        # x: (B, N, d_model)
        # Linear projections (manually, to be able to return grads for weights)
        q = x.matmul(wq_w.t())
        if wq_b is not None:
            q = q + wq_b.view(1, 1, -1)

        k = x.matmul(wk_w.t())
        if wk_b is not None:
            k = k + wk_b.view(1, 1, -1)

        v = x.matmul(wv_w.t())
        if wv_b is not None:
            v = v + wv_b.view(1, 1, -1)

        # call optimized forward pass (returns output, l, m)
        output_attn, l, m, masks = forward_pass(q, k, v, tau, mask, block_sparse_mask)  # (B, N, d_model), l,m shapes
        ctx.masks = masks

        # final linear output projection
        out = output_attn.matmul(wo_w.t())
        if wo_b is not None:
            out = out + wo_b.view(1, 1, -1)

        # save for backward
        ctx.save_for_backward(x, q, k, v, output_attn, wq_w, wk_w, wv_w, wo_w)
        # store non-tensor objects / flags on ctx
        ctx.wq_b_exists = (wq_b is not None)
        ctx.wk_b_exists = (wk_b is not None)
        ctx.wv_b_exists = (wv_b is not None)
        ctx.wo_b_exists = (wo_b is not None)
        ctx.wq_b = wq_b
        ctx.wk_b = wk_b
        ctx.wv_b = wv_b
        ctx.wo_b = wo_b

        ctx.forward_pass = forward_pass
        ctx.backward_pass = backward_pass
        ctx.tau = float(tau)
        # mask / block_sparse_mask may be tensors; we keep references but treat as non-differentiable
        ctx.mask = mask
        ctx.block_sparse_mask = block_sparse_mask

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Return gradients for:
        (x,
         wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b,
         forward_pass, backward_pass, tau, mask, block_sparse_mask)
        Non-tensor args -> return None placeholders.
        """
        # restore
        x, q, k, v, output_attn, wq_w, wk_w, wv_w, wo_w = ctx.saved_tensors
        forward_pass = ctx.forward_pass
        backward_pass = ctx.backward_pass
        tau = ctx.tau
        mask = ctx.mask
        block_sparse_mask = ctx.block_sparse_mask

        B, N, d_model = x.shape

        # 1) grads through output linear (w_o)
        # grad_out: (B, N, d_model)
        # dW_o = grad_out_flat^T @ output_attn_flat
        go_flat = grad_out.reshape(-1, d_model)
        out_flat = output_attn.reshape(-1, d_model)
        dWo = go_flat.t().matmul(out_flat)  # (d_model, d_model)
        dbo = go_flat.sum(dim=0) if ctx.wo_b_exists else None

        # d_output_attn = grad_out @ W_o
        d_output_attn = grad_out.matmul(wo_w)  # (B, N, d_model)

        # 2) call optimized backward to compute dQ, dK, dV
        # BackwardPass expects shapes (B, N, d_model) for q,k,v and dO (grad of output_attn)
        masks = getattr(ctx, "masks", None)
        dQ, dK, dV = backward_pass(q, k, v, d_output_attn, tau, mask, block_sparse_mask, masks)
        # dQ/dK/dV are (B, N, d_model)

        # 3) compute gradients for projection weights and for input x:
        x_flat = x.reshape(-1, d_model)                     # (B*N, d_model)
        dQ_flat = dQ.reshape(-1, d_model)
        dK_flat = dK.reshape(-1, d_model)
        dV_flat = dV.reshape(-1, d_model)

        # weight grads: dW = dProj^T @ x_flat
        dWq = dQ_flat.t().matmul(x_flat)
        dWk = dK_flat.t().matmul(x_flat)
        dWv = dV_flat.t().matmul(x_flat)

        # bias grads if present
        dbq = dQ_flat.sum(dim=0) if ctx.wq_b_exists else None
        dbk = dK_flat.sum(dim=0) if ctx.wk_b_exists else None
        dbv = dV_flat.sum(dim=0) if ctx.wv_b_exists else None

        # grads w.r.t input x from each projection: dx = dProj @ W
        dx_q = dQ_flat.matmul(wq_w).reshape(B, N, d_model)
        dx_k = dK_flat.matmul(wk_w).reshape(B, N, d_model)
        dx_v = dV_flat.matmul(wv_w).reshape(B, N, d_model)

        dx = dx_q + dx_k + dx_v  # total input gradient

        # return gradient tuple matching forward signature
        # (x, wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b, forward_pass, backward_pass, tau, mask, block_sparse_mask)
        return (
            dx,
            dWq, dbq,
            dWk, dbk,
            dWv, dbv,
            dWo, dbo,
            None,  # forward_pass (non-tensor)
            None,  # backward_pass (non-tensor)
            None,  # tau (non-tensor / float)
            None,  # mask (non-differentiable here)
            None   # block_sparse_mask (non-differentiable here)
        )


class Embedding(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.emb = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.emb(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len, dropout):
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)
    self.max_len = max_len

    # learned positional embeddings
    self.pos_emb = nn.Embedding(self.max_len, d_model)
    # initialize similar to transformer practice
    nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

  def forward(self, x):
    # x: (B, L, d_model)
    b, l, _ = x.size()
    positions = torch.arange(l, device=x.device).unsqueeze(0)  # (1, L)
    pos = self.pos_emb(positions)                             # (1, L, d_model)
    x = x + pos
    return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len, use_fused_qkv=True, block_size=128):
        super().__init__()

        # single token embedding (use this for input tokens)
        self.token_embedding = Embedding(vocab_size, d_model)
        # learned positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # decoder-only stack
        self.decoder = nn.ModuleList([
            Decoder(d_model, num_heads, d_ff, dropout, use_fused_qkv, block_size)  # Pass d_ff to each Decoder
            for _ in range(num_layers)
        ])

        # language modeling head
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, tgt_mask=None, tau=1.0):
        x = self.token_embedding(input_ids)       # (B, L, d_model)
        x = self.pos_encoding(x)                  # (B, L, d_model)

        for layer in self.decoder:
            x = layer(x, tgt_mask, tau)

        logits = self.fc_out(x)                  # (B, L, vocab_size)
        return logits