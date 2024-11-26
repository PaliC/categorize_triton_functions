import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_ds_kernel(O, Q, S, Z, DO, DQ, DS, DZ, stride_qk_bh, stride_qk_l,
    stride_qk_d, stride_vo_bh, stride_vo_l, stride_vo_d, stride_s_bh,
    stride_s_dk, stride_s_dv, stride_z_bh, stride_dz_bh, B: 'tl.constexpr',
    H: 'tl.constexpr', L: 'tl.constexpr', DK: 'tl.constexpr', DV:
    'tl.constexpr', BL: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'
    ):
    start_kv, start_m, off_bs_head = tl.program_id(0), tl.program_id(1
        ), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = start_kv // NV
    i_v = start_kv % NV
    O_block_ptr = tl.make_block_ptr(base=O + off_bs_head * stride_vo_bh,
        shape=(L, DV), strides=(stride_vo_l, stride_vo_d), offsets=(start_m *
        BL, i_v * BV), block_shape=(BL, BV), order=(1, 0))
    Q_block_ptr = tl.make_block_ptr(base=Q + off_bs_head * stride_qk_bh,
        shape=(L, DK), strides=(stride_qk_l, stride_qk_d), offsets=(start_m *
        BL, i_k * BK), block_shape=(BL, BK), order=(1, 0))
    DO_block_ptr = tl.make_block_ptr(base=DO + off_bs_head * stride_vo_bh,
        shape=(L, DV), strides=(stride_vo_l, stride_vo_d), offsets=(start_m *
        BL, i_v * BV), block_shape=(BL, BV), order=(1, 0))
    S_block_ptr = tl.make_block_ptr(base=S + off_bs_head * stride_s_bh,
        shape=(DK, DV), strides=(stride_s_dk, stride_s_dv), offsets=(i_k *
        BK, i_v * BV), block_shape=(BK, BV), order=(1, 0))
    Z_block_ptr = Z + off_bs_head * stride_z_bh + i_k * BK + tl.arange(0, BK)
    ds = tl.zeros([BK, BV], dtype=tl.float32)
    dz = tl.zeros([BL], dtype=tl.float32)
    dq = tl.zeros([BL, BK], dtype=tl.float32)
    do = tl.load(DO_block_ptr, boundary_check=(0, 1))
    o = tl.load(O_block_ptr, boundary_check=(0, 1))
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    s = tl.load(S_block_ptr, boundary_check=(0, 1))
    z = tl.load(Z_block_ptr, mask=i_k * BK + tl.arange(0, BK) < DK)
    z = tl.sum(q * z[None, :], axis=1, keep_dims=True) + 1e-06
    ds += tl.dot(tl.trans(q / z), do, allow_tf32=False)
    dz -= tl.sum(o * do / z, axis=1)
    dq += tl.dot(do, tl.trans(s), allow_tf32=False) / z
    DS_block_ptr = tl.make_block_ptr(base=DS + (off_bs_head + B * H *
        start_m) * stride_s_bh, shape=(BK, BV), strides=(stride_s_dk,
        stride_s_dv), offsets=(i_k * BK, i_v * BV), block_shape=(BK, BV),
        order=(1, 0))
    tl.store(DS_block_ptr, ds, boundary_check=(0, 1))
    DQ_block_ptr = tl.make_block_ptr(base=DQ + off_bs_head * stride_vo_bh,
        shape=(L, BK), strides=(stride_vo_l, stride_vo_d), offsets=(start_m *
        BL, i_k * BK), block_shape=(BL, BK), order=(1, 0))
    tl.store(DQ_block_ptr, dq, boundary_check=(0, 1))
    DZ_block_ptr = DZ + off_bs_head * stride_dz_bh + start_m * BL + tl.arange(
        0, BL)
    tl.store(DZ_block_ptr, dz, mask=start_m * BL + tl.arange(0, BL) < L)
