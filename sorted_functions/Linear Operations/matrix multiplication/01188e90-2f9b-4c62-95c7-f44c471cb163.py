import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BT', 'BK', 'BV'])
@triton.jit
def fwd_recompute_w_u_kernel(k, v, beta, w, u, A, T: 'tl.constexpr', H:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', HEAD_FIRST:
    'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if HEAD_FIRST:
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,),
            (BT,), (0,))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT, 0), (BT, BT), (1, 0))
    else:
        p_beta = tl.make_block_ptr(beta + i_b * T * H + i_h, (T,), (H,), (
            i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(A + i_b * T * H * BT + i_h * BT, (T, BT), (
            H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t *
                BT, i_v * BV), (BT, BV), (1, 0))
            p_u = tl.make_block_ptr(u + i_bh * T * V, (T, V), (V, 1), (i_t *
                BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_u = tl.make_block_ptr(u + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = b_v * b_beta[:, None]
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u, boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT, i_k * BK), (BT, BK), (1, 0))
            p_w = tl.make_block_ptr(w + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_w = tl.make_block_ptr(w + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_w = tl.dot(b_A, b_kb, allow_tf32=False)
        tl.store(p_w, b_w, boundary_check=(0, 1))
