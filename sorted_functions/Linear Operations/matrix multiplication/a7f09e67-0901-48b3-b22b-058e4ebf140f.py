import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BL': 128, 'BK': 128, 'BV': 128},
    num_warps=8), triton.Config({'BL': 128, 'BK': 64, 'BV': 64}, num_warps=
    4), triton.Config({'BL': 64, 'BK': 64, 'BV': 64}, num_warps=2)], key=[
    'L', 'DK', 'DV'])
@triton.jit
def _fwd_kv_kernel(K, V, S, Z, stride_qk_bh, stride_qk_l, stride_qk_d,
    stride_vo_bh, stride_vo_l, stride_vo_d, stride_s_bh, stride_s_dk,
    stride_s_dv, stride_z_bh, scale, B: 'tl.constexpr', H: 'tl.constexpr',
    L: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', BL:
    'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    start_v, start_k, off_bs_head = tl.program_id(0), tl.program_id(1
        ), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    K_block_ptr = tl.make_block_ptr(base=K + off_bs_head * stride_qk_bh,
        shape=(DK, L), strides=(stride_qk_d, stride_qk_l), offsets=(start_k *
        BK, 0), block_shape=(BK, BL), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + off_bs_head * stride_vo_bh,
        shape=(L, DV), strides=(stride_vo_l, stride_vo_d), offsets=(0, 
        start_v * BV), block_shape=(BL, BV), order=(1, 0))
    s = tl.zeros([BK, BV], dtype=tl.float32)
    z = tl.zeros([BK], dtype=tl.float32)
    for _ in range(0, L, BL):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
        v = v * scale
        s += tl.dot(k, v, allow_tf32=False)
        z += tl.sum(k, axis=1) / L
        K_block_ptr = tl.advance(K_block_ptr, (0, BL))
        V_block_ptr = tl.advance(V_block_ptr, (BL, 0))
    S_block_ptr = tl.make_block_ptr(base=S + off_bs_head * stride_s_bh,
        shape=(DK, DV), strides=(stride_s_dk, stride_s_dv), offsets=(
        start_k * BK, start_v * BV), block_shape=(BK, BV), order=(1, 0))
    tl.store(S_block_ptr, s, boundary_check=(0, 1))
    Z_block_ptr = Z + off_bs_head * stride_z_bh + start_k * BK + tl.arange(
        0, BK)
    tl.store(Z_block_ptr, z, mask=start_k * BK + tl.arange(0, BK) < DK)
