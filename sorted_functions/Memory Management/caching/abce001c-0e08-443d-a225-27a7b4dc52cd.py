import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_retention_fwd_kernel_h(k, v, h, h0, ht, H: 'tl.constexpr', T:
    'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT:
    'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE:
    'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1))
    for i_t in range(NT):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (i_k *
                BK, i_t * BT), (BK, BT), (0, 1))
            p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t *
                BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (K, T),
                (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_v = tl.make_block_ptr(v + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * NT * K * V + i_t * K * V, (K, V),
            (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if i_t == NT - 1 and T % BT != 0:
            d_b = tl.math.exp2(T % BT * b_b)
            d_i = tl.math.exp2((T % BT - o_i - 1) * b_b)
        b_h = d_b * b_h + tl.dot(b_k, b_v * d_i[:, None], allow_tf32=False)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))
