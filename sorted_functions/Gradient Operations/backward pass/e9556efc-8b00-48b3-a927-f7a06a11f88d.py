import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_delta_rule_bwd_kernel_dqkw(q, k, v, w, h, do, dh, dq, dk, dv, dw,
    scale, T: 'tl.constexpr', H: 'tl.constexpr', K: 'tl.constexpr', V:
    'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV:
    'tl.constexpr', NT: 'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BT)
    if HEAD_FIRST:
        p_q = tl.make_block_ptr(q + i_bh * T * K, (K, T), (1, K), (i_k * BK,
            i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT,
            i_k * BK), (BT, BK), (1, 0))
    else:
        p_q = tl.make_block_ptr(q + i_b * T * H * K + i_h * K, (K, T), (1, 
            H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t *
                BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (
                i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + i_bh * T * V, (T, V), (V, 1), (
                i_t * BT, i_v * BV), (BT, BV), (1, 0))
        else:
            p_v = tl.make_block_ptr(v + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * NT * K * V, (V, NT * K), (1, V),
            (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + i_bh * NT * K * V, (V, NT * K), (1, V
            ), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, b_dh, allow_tf32=False)
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_dw += tl.dot(b_dv, b_h, allow_tf32=False)
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds, 0)
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dq *= scale
    b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
    if HEAD_FIRST:
        p_dq = tl.make_block_ptr(dq + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + i_bh * T * K, (T, K), (K, 1), (i_t *
            BT, i_k * BK), (BT, BK), (1, 0))
    else:
        p_dq = tl.make_block_ptr(dq + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + i_b * T * H * K + i_h * K, (T, K), (H *
            K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dw, -b_dw, boundary_check=(0, 1))
