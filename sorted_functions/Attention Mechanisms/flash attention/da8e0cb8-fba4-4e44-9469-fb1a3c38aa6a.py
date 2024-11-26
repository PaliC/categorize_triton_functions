import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BL': 128}, num_warps=8), triton.
    Config({'BL': 128}, num_warps=4), triton.Config({'BL': 128}, num_warps=
    2), triton.Config({'BL': 64}, num_warps=8), triton.Config({'BL': 64},
    num_warps=4), triton.Config({'BL': 64}, num_warps=2)], key=['L'])
@triton.jit
def _fwd_qs_kernel(Q, S, O, Z, stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d, stride_s_bh, stride_s_dk,
    stride_s_dv, stride_z_bh, B: 'tl.constexpr', H: 'tl.constexpr', L:
    'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', BL:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    start_v, start_m, off_bs_head = tl.program_id(0), tl.program_id(1
        ), tl.program_id(2)
    qkv_base_offset = off_bs_head * stride_qk_bh
    NV = tl.cdiv(DV, BV)
    Q_block_ptr = tl.make_block_ptr(base=Q + off_bs_head * stride_qk_bh,
        shape=(L, DK), strides=(stride_qk_l, stride_qk_d), offsets=(start_m *
        BL, 0), block_shape=(BL, BK), order=(1, 0))
    S_block_ptr = tl.make_block_ptr(base=S + off_bs_head * stride_s_bh,
        shape=(DK, DV), strides=(stride_s_dk, stride_s_dv), offsets=(0, 
        start_v * BV), block_shape=(BK, BV), order=(1, 0))
    Z_block_ptr = Z + off_bs_head * stride_z_bh + tl.arange(0, BK)
    o = tl.zeros([BL, BV], dtype=tl.float32)
    z_buffer = tl.zeros([BL], dtype=tl.float32)
    for i_k in range(0, DK, BK):
        q = tl.load(Q_block_ptr, boundary_check=(0, 1))
        s = tl.load(S_block_ptr, boundary_check=(0, 1))
        z = tl.load(Z_block_ptr, mask=i_k + tl.arange(0, BK) < DK)
        z_buffer += tl.sum(q * z[None, :], axis=1, keep_dims=False)
        o += tl.dot(q, s, allow_tf32=False)
        Q_block_ptr = tl.advance(Q_block_ptr, (0, BK))
        S_block_ptr = tl.advance(S_block_ptr, (BK, 0))
        Z_block_ptr = Z_block_ptr + tl.arange(0, BK)
    o = o / (z_buffer[:, None] + 1e-06)
    O_block_ptr = tl.make_block_ptr(base=O + off_bs_head * stride_vo_bh,
        shape=(L, DV), strides=(stride_vo_l, stride_vo_d), offsets=(start_m *
        BL, start_v * BV), block_shape=(BL, BV), order=(1, 0))
    tl.store(O_block_ptr, o, boundary_check=(0, 1))
