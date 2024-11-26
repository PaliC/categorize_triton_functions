import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4)], key=['BT'])
@triton.jit
def compute_final_dg(dg, o, T: 'tl.constexpr', BT: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_o = tl.make_block_ptr(dg + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    b_o = tl.load(p_o, boundary_check=(0,))
    b_o = b_o - tl.cumsum(b_o, axis=0) + tl.sum(b_o, axis=0)
    p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_o, b_o, boundary_check=(0,))
