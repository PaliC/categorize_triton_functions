import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BT'])
@triton.jit
def chunk_local_cumsum_scalar_kernel(s, o, T: 'tl.constexpr', BT:
    'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,))
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o, boundary_check=(0,))
