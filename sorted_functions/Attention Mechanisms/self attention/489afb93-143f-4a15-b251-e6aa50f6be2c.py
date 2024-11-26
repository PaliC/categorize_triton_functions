import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args[
    'BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args[
    'BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM']})
@triton.jit
def _fwd_attn_kernel(Q, K, V, B, softmax_scale, stride_qb, stride_qh,
    stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh,
    stride_vn, stride_bb, stride_bh, stride_bm, stride_bn, stride_ob,
    stride_oh, stride_om, stride_lb, stride_lh, headdim, seqlen_q, seqlen_k,
    O, L, HAVE_BIAS: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_M:
    'tl.constexpr', EVEN_N: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr',
    BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m, off_b, off_h = tl.program_id(0), tl.program_id(1), tl.program_id(2
        )
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_n = tl.arange(0, BLOCK_N)
    q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] *
        stride_qm + offs_d[None, :]))
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    elif EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[
            None, :] < headdim), other=0.0)
    k_ptrs = K + (off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] *
        stride_kn + offs_d[None, :]))
    v_ptrs = V + (off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] *
        stride_vn + offs_d[None, :]))
    softmax_scale = softmax_scale
    if HAVE_BIAS:
        b_ptrs = B + (off_b * stride_bb + off_h * stride_bh + (offs_m[:,
            None] * stride_bm + offs_n[None, :]))
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    max_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    for j in range(0, seqlen_k, BLOCK_N):
        j = tl.multiple_of(j, BLOCK_N)
        if EVEN_N:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + j * stride_kn)
            else:
                k = tl.load(k_ptrs + j * stride_kn, mask=offs_d[None, :] <
                    headdim, other=0.0)
        elif EVEN_HEADDIM:
            k = tl.load(k_ptrs + j * stride_kn, mask=(j + offs_n)[:, None] <
                seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs + j * stride_kn, mask=((j + offs_n)[:, None] <
                seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k.T)
        if not EVEN_N:
            qk += tl.where((j + offs_n)[None, :] < seqlen_k, 0, float('-inf'))
        if HAVE_BIAS:
            if EVEN_N & EVEN_M:
                b = tl.load(b_ptrs + j)
            else:
                b = tl.load(b_ptrs + j, mask=(offs_m[:, None] < seqlen_q) &
                    (j + offs_n)[None, :] < seqlen_k, other=0.0)
            qk = qk * softmax_scale + b
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        else:
            max_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - max_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(max_i - max_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        if EVEN_N:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + j * stride_vn)
            else:
                v = tl.load(v_ptrs + j * stride_vn, mask=offs_d[None, :] <
                    headdim, other=0.0)
        elif EVEN_HEADDIM:
            v = tl.load(v_ptrs + j * stride_vn, mask=(j + offs_n)[:, None] <
                seqlen_k, other=0.0)
        else:
            v = tl.load(v_ptrs + j * stride_vn, mask=((j + offs_n)[:, None] <
                seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        acc_o += tl.dot(p, v)
        max_i = max_ij
        lin = tl.exp(lse_i - max_ij) + l_ij
        lse_i = max_ij + tl.log(lin)
    o_scale = tl.exp(max_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    lse_ptrs = L + (off_b * stride_lb + off_h * stride_lh + offs_m)
    tl.store(lse_ptrs, lse_i, mask=offs_m < seqlen_q)
    out_ptrs = O + (off_b * stride_ob + off_h * stride_oh + (offs_m[:, None
        ] * stride_om + offs_d[None, :]))
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    elif EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
    else:
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (
            offs_d[None, :] < headdim))
