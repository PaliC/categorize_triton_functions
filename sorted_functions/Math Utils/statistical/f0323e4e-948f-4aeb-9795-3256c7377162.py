import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BS': 16}, num_warps=2), triton.
    Config({'BS': 16}, num_warps=4), triton.Config({'BS': 16}, num_warps=8),
    triton.Config({'BS': 32}, num_warps=2), triton.Config({'BS': 32},
    num_warps=4), triton.Config({'BS': 32}, num_warps=8), triton.Config({
    'BS': 64}, num_warps=2), triton.Config({'BS': 64}, num_warps=4), triton
    .Config({'BS': 64}, num_warps=8)], key=['S', 'BT'])
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_local_cumsum_vector_kernel(s, o, offsets, T: 'tl.constexpr', H:
    'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS:
    'tl.constexpr', HEAD_FIRST: 'tl.constexpr', USE_OFFSETS: 'tl.constexpr'):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b), tl.load(offsets + i_b + 1)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + i_bh * T * S, (T, S), (S, 1), (i_t * BT,
            i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * T * S, (T, S), (S, 1), (i_t * BT,
            i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + start * H * S + i_h * S, (T, S), (H * S,
            1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + start * H * S + i_h * S, (T, S), (H * S,
            1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o, boundary_check=(0, 1))
