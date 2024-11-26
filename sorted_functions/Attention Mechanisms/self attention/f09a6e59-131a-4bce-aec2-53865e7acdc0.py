import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kernel(Q, K, V, sm_scale, Out, DO, DQ, DK, DV, L, D, stride_dqa,
    stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
    stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, Z, H,
    N_CTX, Z_H_N_CTX, SQ_Z_H_N_CTX, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', SEQUENCE_PARALLEL:
    'tl.constexpr', CAUSAL: 'tl.constexpr', MMA_V3: 'tl.constexpr'):
    qk_scale = sm_scale * 1.44269504
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    Q_block_ptr = tl.make_block_ptr(base=Q, shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk), offsets=(0, 0), block_shape=(
        BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K, shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk), offsets=(0, 0), block_shape=(
        BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=V, shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk), offsets=(0, 0), block_shape=(
        BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    DO_block_ptr = tl.make_block_ptr(base=DO, shape=(Z_H_N_CTX,
        BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    if SEQUENCE_PARALLEL:
        DQ_block_ptr = tl.make_block_ptr(base=DQ, shape=(SQ_Z_H_N_CTX,
            BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    else:
        DQ_block_ptr = tl.make_block_ptr(base=DQ, shape=(Z_H_N_CTX,
            BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    DK_block_ptr = tl.make_block_ptr(base=DK, shape=(Z_H_N_CTX,
        BLOCK_DMODEL), strides=(stride_kn, stride_kk), offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    DV_block_ptr = tl.make_block_ptr(base=DV, shape=(Z_H_N_CTX,
        BLOCK_DMODEL), strides=(stride_vn, stride_vk), offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    if not SEQUENCE_PARALLEL:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale, Out, DO,
                DQ, DK, DV, L, D, Q_block_ptr, K_block_ptr, V_block_ptr,
                DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,
                stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,
                stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                stride_vh, stride_vn, stride_vk, Z, H, N_CTX, off_h, off_z,
                off_hz, start_n, num_block_n, BLOCK_M=BLOCK_M, BLOCK_DMODEL
                =BLOCK_DMODEL, BLOCK_N=BLOCK_N, SEQUENCE_PARALLEL=
                SEQUENCE_PARALLEL, CAUSAL=CAUSAL, MMA_V3=MMA_V3)
    else:
        start_n = tl.program_id(1)
        _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale, Out, DO, DQ,
            DK, DV, L, D, Q_block_ptr, K_block_ptr, V_block_ptr,
            DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,
            stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
            stride_vh, stride_vn, stride_vk, Z, H, N_CTX, off_h, off_z,
            off_hz, start_n, num_block_n, BLOCK_M=BLOCK_M, BLOCK_DMODEL=
            BLOCK_DMODEL, BLOCK_N=BLOCK_N, SEQUENCE_PARALLEL=
            SEQUENCE_PARALLEL, CAUSAL=CAUSAL, MMA_V3=MMA_V3)
