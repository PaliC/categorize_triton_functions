import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.
    Config({'BT': 32}, num_warps=4), triton.Config({'BT': 32}, num_warps=2),
    triton.Config({'BT': 64}, num_warps=8), triton.Config({'BT': 64},
    num_warps=4)], key=[])
@triton.jit
def chunk_global_cumsum_scalar_kernel(s, o, T: 'tl.constexpr', BT:
    'tl.constexpr'):
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,
            ), (0,))
        p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,
            ), (0,))
        b_s = tl.load(p_s, boundary_check=(0,))
        b_o = tl.cumsum(b_s, axis=0) + b_z[None]
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        tl.store(p_o, b_o, boundary_check=(0,))
