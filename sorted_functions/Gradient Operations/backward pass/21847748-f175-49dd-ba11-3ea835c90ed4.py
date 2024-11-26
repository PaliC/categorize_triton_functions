import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_dkv_kernel(K, V, dK, dV, dS, stride_qk_bh, stride_qk_l,
    stride_qk_d, stride_s_bh, stride_s_dk, stride_s_dv, scale, B:
    'tl.constexpr', H: 'tl.constexpr', L: 'tl.constexpr', DK:
    'tl.constexpr', DV: 'tl.constexpr', BL: 'tl.constexpr', BK:
    'tl.constexpr', BV: 'tl.constexpr'):
    start_kv, start_m, off_bs_head = tl.program_id(0), tl.program_id(1
        ), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = start_kv // NV
    i_v = start_kv % NV
    qkv_base_offset = off_bs_head * stride_qk_bh
    K_block_ptr = tl.make_block_ptr(base=K + qkv_base_offset, shape=(L, DK),
        strides=(stride_qk_l, stride_qk_d), offsets=(start_m * BL, 0),
        block_shape=(BL, BK), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=V + qkv_base_offset, shape=(L, DV),
        strides=(stride_qk_l, stride_qk_d), offsets=(start_m * BL, 0),
        block_shape=(BL, BV), order=(1, 0))
    dS_block_ptr = tl.make_block_ptr(base=dS + off_bs_head * stride_s_bh,
        shape=(DK, DV), strides=(stride_s_dk, stride_s_dv), offsets=(0, 0),
        block_shape=(BK, BV), order=(1, 0))
    dk = tl.zeros([BL, BK], dtype=tl.float32)
    dv = tl.zeros([BL, BV], dtype=tl.float32)
    ds = tl.load(dS_block_ptr, boundary_check=(0, 1))
    k = tl.load(K_block_ptr, boundary_check=(0, 1))
    v = tl.load(V_block_ptr, boundary_check=(0, 1))
    v = v * scale
    dk += tl.dot(v, tl.trans(ds), allow_tf32=False)
    dv += tl.dot(k, ds, allow_tf32=False) * scale
    dK_block_ptr = tl.make_block_ptr(base=dK + qkv_base_offset, shape=(L,
        DK), strides=(stride_qk_l, stride_qk_d), offsets=(start_m * BL, 0),
        block_shape=(BL, BK), order=(1, 0))
    dV_block_ptr = tl.make_block_ptr(base=dV + qkv_base_offset, shape=(L,
        DV), strides=(stride_qk_l, stride_qk_d), offsets=(start_m * BL, 0),
        block_shape=(BL, BV), order=(1, 0))
    tl.store(dK_block_ptr, dk, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dv, boundary_check=(0, 1))
