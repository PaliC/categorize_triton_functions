import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BV', 'BT'])
@triton.jit
def chunk_gla_bwd_kernel_dA(v, do, dA, scale, T: 'tl.constexpr', H:
    'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BV:
    'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_do = tl.make_block_ptr(do + i_bh * T * V, (T, V), (V, 1), (
                i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_v = tl.make_block_ptr(v + i_bh * T * V, (V, T), (1, V), (i_v *
                BV, i_t * BT), (BV, BT), (0, 1))
        else:
            p_do = tl.make_block_ptr(do + i_b * T * H * V + i_h * V, (T, V),
                (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_v = tl.make_block_ptr(v + i_b * T * H * V + i_h * V, (V, T),
                (1, H * V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dA += tl.dot(b_do, b_v)
    if HEAD_FIRST:
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT, 0), (BT, BT), (1, 0))
    else:
        p_dA = tl.make_block_ptr(dA + i_b * T * H * BT + i_h * BT, (T, BT),
            (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA = tl.where(m_s, b_dA * scale, 0.0)
    tl.store(p_dA, b_dA, boundary_check=(0, 1))
