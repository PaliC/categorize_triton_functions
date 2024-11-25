import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_N': lambda args: args['seqlen_k'] % args[
    'BLOCK_N'] == 0})
@triton.jit
def _fwd_attn_kernel_block_ptr(Q, K, V, B, softmax_scale: 'tl.constexpr',
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
    if not EVEN_N:
        offs_n = tl.arange(0, BLOCK_N)
    Q_Block_ptr = tl.make_block_ptr(base=Q + (off_b * stride_qb + off_h *
        stride_qh), shape=(seqlen_q, headdim), block_shape=(BLOCK_M,
        BLOCK_HEADDIM), strides=(stride_qm, 1), offsets=(start_m * BLOCK_M,
        0), order=(0, 1))
    O_Block_ptr = tl.make_block_ptr(base=O + (off_b * stride_ob + off_h *
        stride_oh), shape=(seqlen_q, headdim), block_shape=(BLOCK_M,
        BLOCK_HEADDIM), strides=(stride_om, 1), offsets=(start_m * BLOCK_M,
        0), order=(0, 1))
    L_Block_ptr = tl.make_block_ptr(base=L + (off_b * stride_lb + off_h *
        stride_lh), shape=(seqlen_q,), strides=(1,), offsets=(start_m *
        BLOCK_M,), block_shape=(BLOCK_M,), order=(0,))
    kv_stride = off_b * stride_kb + off_h * stride_kh
    K_Block_ptr = tl.make_block_ptr(base=K + kv_stride, shape=(headdim,
        seqlen_k), block_shape=(BLOCK_HEADDIM, BLOCK_N), strides=(1,
        stride_kn), offsets=(0, 0), order=(1, 0))
    V_Block_ptr = tl.make_block_ptr(base=V + kv_stride, shape=(seqlen_k,
        headdim), block_shape=(BLOCK_N, BLOCK_HEADDIM), strides=(stride_vn,
        1), offsets=(0, 0), order=(0, 1))
    q = tl.load(Q_Block_ptr, boundary_check=(0, 1))
    softmax_scale = softmax_scale
    if HAVE_BIAS:
        bias_h_pos: 'tl.constexpr' = 0 if BIAS_SINGLE_HEAD else off_h
        B_Block_ptr = tl.make_block_ptr(base=B + (off_b * stride_bb + 
            bias_h_pos * stride_bh), shape=(seqlen_q, seqlen_k),
            block_shape=(BLOCK_M, BLOCK_N), strides=(stride_bm, stride_bn),
            offsets=(start_m * BLOCK_M, 0), order=(0, 1))
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    max_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    for j in range(0, seqlen_k, BLOCK_N):
        j = tl.multiple_of(j, BLOCK_N)
        k = tl.load(K_Block_ptr, boundary_check=(0, 1))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k) * softmax_scale
        if not EVEN_N:
            qk += tl.where((j + offs_n)[None, :] < seqlen_k, 0, float('-inf'))
        if HAVE_BIAS:
            b = tl.load(B_Block_ptr, boundary_check=(0, 1))
            B_Block_ptr = tl.advance(B_Block_ptr, (0, BLOCK_N))
            qk = qk + b
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        else:
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(max_i - max_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v = tl.load(V_Block_ptr, boundary_check=(0, 1))
        acc_o += tl.dot(p, v)
        max_i = max_ij
        lse_i = max_ij + tl.log(tl.exp(lse_i - max_ij) + l_ij)
        K_Block_ptr = tl.advance(K_Block_ptr, (0, BLOCK_N))
        V_Block_ptr = tl.advance(V_Block_ptr, (BLOCK_N, 0))
    o_scale = tl.exp(max_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    tl.store(L_Block_ptr, lse_i, boundary_check=(0,))
    tl.store(O_Block_ptr, acc_o, boundary_check=(0, 1))
