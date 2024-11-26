import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.
    Config({'BT': 32}, num_warps=4), triton.Config({'BT': 32}, num_warps=2),
    triton.Config({'BT': 64}, num_warps=8), triton.Config({'BT': 64},
    num_warps=4)], key=[])
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(s, o, offsets, T:
    'tl.constexpr', H: 'tl.constexpr', BT: 'tl.constexpr', HEAD_FIRST:
    'tl.constexpr', USE_OFFSETS: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        start, end = tl.load(offsets + i_b), tl.load(offsets + i_b + 1)
    else:
        start, end = i_b * T, i_b * T + T
    T = end - start
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,),
                (BT,), (0,))
            p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,),
                (BT,), (0,))
        else:
            p_s = tl.make_block_ptr(s + start * H + i_h, (T,), (H,), (i_t *
                BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + start * H + i_h, (T,), (H,), (i_t *
                BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,))
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
        tl.store(p_o, b_o, boundary_check=(0,))
