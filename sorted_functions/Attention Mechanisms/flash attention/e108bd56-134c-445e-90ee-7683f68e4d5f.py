import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_delta_rule_fwd_kernel_prepare_dv(q, k, do, dv, scale, T:
    'tl.constexpr', H: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr',
    BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', HEAD_FIRST:
    'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_q = tl.make_block_ptr(q + i_bh * T * K, (K, T), (1, K), (i_k *
                BK, i_t * BT), (BK, BT), (0, 1))
            p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_q = tl.make_block_ptr(q + i_b * T * H * K + i_h * K, (K, T),
                (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_A += tl.dot(b_k, b_q, allow_tf32=False)
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :],
        b_A, 0)
    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (
                i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + i_bh * T * V, (T, V), (V, 1), (
                i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_do = tl.make_block_ptr(do + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
