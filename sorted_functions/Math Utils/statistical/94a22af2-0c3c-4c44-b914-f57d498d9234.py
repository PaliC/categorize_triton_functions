import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.
    Config({'BT': 16}, num_warps=4), triton.Config({'BT': 16}, num_warps=8),
    triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 32},
    num_warps=4), triton.Config({'BT': 32}, num_warps=8), triton.Config({
    'BT': 64}, num_warps=2), triton.Config({'BT': 64}, num_warps=4), triton
    .Config({'BT': 64}, num_warps=8)], key=['S'])
@triton.jit
def chunk_cumsum_fwd_kernel(s, z, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S:
    'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr'):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
            i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (
            i_t * BT, i_s * BS), (BT, BS), (1, 0))
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c, boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_s, 0)
