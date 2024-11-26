import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BS': 16}, num_warps=2), triton.
    Config({'BS': 16}, num_warps=4), triton.Config({'BS': 16}, num_warps=8),
    triton.Config({'BS': 32}, num_warps=2), triton.Config({'BS': 32},
    num_warps=4), triton.Config({'BS': 32}, num_warps=8), triton.Config({
    'BS': 64}, num_warps=2), triton.Config({'BS': 64}, num_warps=4), triton
    .Config({'BS': 64}, num_warps=8)], key=['S'])
@triton.jit
def chunk_rwkv6_fwd_cumsum_kernel(s, o, o_minus_s, s_s_h, s_s_t, s_s_d, T:
    'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr'):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t *
        BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t *
        BT, i_s * BS), (BT, BS), (1, 0))
    p_o_minus_s = tl.make_block_ptr(o_minus_s + i_bh * s_s_h, (T, S), (
        s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o, boundary_check=(0, 1))
    tl.store(p_o_minus_s, b_o - b_s, boundary_check=(0, 1))
