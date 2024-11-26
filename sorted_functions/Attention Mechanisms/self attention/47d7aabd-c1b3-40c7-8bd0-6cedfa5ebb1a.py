import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_delta_rule_fwd_kernel_h(k, v, d, v_new, h, h0, ht, T:
    'tl.constexpr', H: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr',
    BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', NT: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr',
    STORE_FINAL_STATE: 'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1))
    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_bh * NT * K * V + i_t * K * V, (K, V),
            (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            if HEAD_FIRST:
                p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (
                    i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_d = tl.make_block_ptr(d + i_bh * T * K, (T, K), (K, 1), (
                    i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (
                    i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new + i_bh * T * V, (T, V), (
                    V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            else:
                p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (K,
                    T), (1, H * K), (i_k * BK, i_t * BT + i_c * BC), (BK,
                    BC), (0, 1))
                p_d = tl.make_block_ptr(d + i_b * T * H * K + i_h * K, (T,
                    K), (H * K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC,
                    BK), (1, 0))
                p_v = tl.make_block_ptr(v + i_b * T * H * V + i_h * V, (T,
                    V), (H * V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC,
                    BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new + i_b * T * H * V + i_h *
                    V, (T, V), (H * V, 1), (i_t * BT + i_c * BC, i_v * BV),
                    (BC, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_v -= tl.dot(b_d, b_h)
            tl.store(p_v_new, b_v, boundary_check=(0, 1))
            b_hc += tl.dot(b_k, b_v, allow_tf32=False)
        b_h += b_hc
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))
