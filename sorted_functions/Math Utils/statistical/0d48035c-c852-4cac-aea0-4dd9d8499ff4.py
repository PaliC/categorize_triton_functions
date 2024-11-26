import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.
    Config({'BT': 16}, num_warps=4), triton.Config({'BT': 16}, num_warps=8),
    triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 32},
    num_warps=4), triton.Config({'BT': 32}, num_warps=8), triton.Config({
    'BT': 64}, num_warps=2), triton.Config({'BT': 64}, num_warps=4), triton
    .Config({'BT': 64}, num_warps=8)], key=['S'])
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_global_cumsum_vector_kernel(s, z, offsets, T: 'tl.constexpr', H:
    'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS:
    'tl.constexpr', HEAD_FIRST: 'tl.constexpr', USE_OFFSETS: 'tl.constexpr'):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b), tl.load(offsets + i_b + 1)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + i_bh * T * S, (T, S), (S, 1), (i_t *
                BT, i_s * BS), (BT, BS), (1, 0))
            p_z = tl.make_block_ptr(z + i_bh * T * S, (T, S), (S, 1), (i_t *
                BT, i_s * BS), (BT, BS), (1, 0))
        else:
            p_s = tl.make_block_ptr(s + start * H * S + i_h * S, (T, S), (H *
                S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_z = tl.make_block_ptr(z + start * H * S + i_h * S, (T, S), (H *
                S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c, boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_s, 0)
