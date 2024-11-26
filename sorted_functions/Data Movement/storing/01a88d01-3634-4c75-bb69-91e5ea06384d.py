import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2)], key=['BT'])
@triton.jit
def save_intra_chunk_attn(A, A_local, T: 'tl.constexpr', BT: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_A = tl.make_block_ptr(A + i_bh * T * T, (T, T), (T, 1), (i_t * BT, 
        i_t * BT), (BT, BT), (1, 0))
    p_A_local = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1),
        (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_local = tl.load(p_A_local, boundary_check=(0, 1))
    tl.store(p_A, b_A_local, boundary_check=(0, 1))
