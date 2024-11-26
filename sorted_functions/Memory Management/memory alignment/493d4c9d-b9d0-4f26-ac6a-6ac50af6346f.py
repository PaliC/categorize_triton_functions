import triton
import triton.language as tl
import torch

@triton.jit
def chunk_abc_bwd_kernel_rcum_inter(c, z, s_s_h, s_s_t, s_s_d, T:
    'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS:
    'tl.constexpr', NT: 'tl.constexpr'):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_c = tl.make_block_ptr(c + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
            i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * NT * S, (NT * S,), (s_s_d,), (
            i_t * S + i_s * BS,), (BS,), (0,))
        b_c = tl.load(p_c, boundary_check=(0, 1)) + b_z
        b_z += tl.load(p_z, boundary_check=(0,))
        tl.store(p_c, b_c, boundary_check=(0, 1))
