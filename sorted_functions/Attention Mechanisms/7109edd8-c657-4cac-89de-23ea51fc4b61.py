import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_N': lambda args: args['seqlen_k'] % args[
    'BLOCK_N'] == 0})
@triton.jit
def _fwd_attn_kernel_ptr_block(Q, K, V, B, softmax_scale: 'tl.constexpr',
    stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm,
    stride_bn, stride_ob, stride_oh, stride_om, stride_lb, stride_lh,
    headdim: 'tl.constexpr', nheads: 'tl.constexpr', seqlen_q, seqlen_k, O,
    L, HAVE_BIAS: 'tl.constexpr', BIAS_SINGLE_HEAD: 'tl.constexpr',
    BLOCK_HEADDIM: 'tl.constexpr', EVEN_N: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m, off_bh = tl.program_id(0), tl.program_id(1)
    off_h = off_bh % nheads
    off_b = off_bh // nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh) + (offs_m[:, None] *
        stride_qm + offs_d[None, :])
    o_ptrs = O + (off_b * stride_ob + off_h * stride_oh) + (offs_m[:, None] *
        stride_om + offs_d[None, :])
    l_ptrs = L + (off_b * stride_lb + off_h * stride_lh + offs_m)
    k_ptrs = K + (off_b * stride_kb + off_h * stride_kh) + (offs_n[:, None] *
        stride_kn + offs_d[None, :])
    v_ptrs = V + (off_b * stride_vb + off_h * stride_vh) + (offs_n[:, None] *
        stride_vn + offs_d[None, :])
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :
        ] < headdim), other=0.0)
    softmax_scale = softmax_scale
    if HAVE_BIAS:
        bias_h_pos: 'tl.constexpr' = 0 if BIAS_SINGLE_HEAD else off_h
        b_ptrs = B + (off_b * stride_bb + bias_h_pos * stride_bh) + (offs_m
            [:, None] * stride_bm + offs_n[None, :] * stride_bn)
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    max_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    for j in range(0, seqlen_k, BLOCK_N):
        j = tl.multiple_of(j, BLOCK_N)
        current_k = offs_n + j
        k = tl.load(k_ptrs + j * stride_kn, mask=(current_k[:, None] <
            seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k.T) * softmax_scale
        if not EVEN_N:
            qk += tl.where((j + offs_n)[None, :] < seqlen_k, 0, float('-inf'))
        if HAVE_BIAS:
            b = tl.load(b_ptrs + j, mask=(offs_m[:, None] < seqlen_q) & (
                current_k[None, :] < seqlen_k), other=0.0)
            qk = qk + b
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        else:
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(max_i - max_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v = tl.load(v_ptrs + j * stride_vn, mask=(current_k[:, None] <
            seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        acc_o += tl.dot(p, v)
        max_i = max_ij
        lse_i = max_ij + tl.log(tl.exp(lse_i - max_ij) + l_ij)
    o_scale = tl.exp(max_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    tl.store(l_ptrs, lse_i, mask=offs_m < seqlen_q)
    tl.store(o_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[
        None, :] < headdim))
