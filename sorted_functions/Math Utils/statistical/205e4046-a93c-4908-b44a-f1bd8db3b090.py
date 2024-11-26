import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BT'])
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_local_cumsum_scalar_kernel(s, o, offsets, T: 'tl.constexpr', H:
    'tl.constexpr', BT: 'tl.constexpr', HEAD_FIRST: 'tl.constexpr',
    USE_OFFSETS: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b), tl.load(offsets + i_b + 1)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start
    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,
            ), (0,))
        p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,
            ), (0,))
    else:
        p_s = tl.make_block_ptr(s + start * H + i_h, (T,), (H,), (i_t * BT,
            ), (BT,), (0,))
        p_o = tl.make_block_ptr(o + start * H + i_h, (T,), (H,), (i_t * BT,
            ), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,))
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o, boundary_check=(0,))