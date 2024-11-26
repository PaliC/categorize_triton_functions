import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale, Out, DO, DQ, DK,
    DV, L, D, Q_block_ptr, K_block_ptr, V_block_ptr, DO_block_ptr,
    DQ_block_ptr, DK_block_ptr, DV_block_ptr, stride_dqa, stride_qz,
    stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn,
    stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, Z, H, N_CTX,
    off_h, off_z, off_hz, start_n, num_block, BLOCK_M: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    SEQUENCE_PARALLEL: 'tl.constexpr', CAUSAL: 'tl.constexpr', MMA_V3:
    'tl.constexpr'):
    if CAUSAL:
        lo = start_n * BLOCK_M
    else:
        lo = 0
    Q_offset = (off_z * stride_qz + off_h * stride_qh) // stride_qm
    DQ_offset = off_z * stride_qz + off_h * stride_qh
    K_offset = (off_z * stride_kz + off_h * stride_kh) // stride_kn
    V_offset = (off_z * stride_vz + off_h * stride_vh) // stride_vn
    if SEQUENCE_PARALLEL:
        DQ_offset += stride_dqa * start_n
    DQ_offset = DQ_offset // stride_qm
    Q_block_ptr = tl.advance(Q_block_ptr, (lo + Q_offset, 0))
    K_block_ptr = tl.advance(K_block_ptr, (start_n * BLOCK_M + K_offset, 0))
    V_block_ptr = tl.advance(V_block_ptr, (start_n * BLOCK_M + V_offset, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo + Q_offset, 0))
    DQ_block_ptr = tl.advance(DQ_block_ptr, (lo + DQ_offset, 0))
    DK_block_ptr = tl.advance(DK_block_ptr, (start_n * BLOCK_M + K_offset, 0))
    DV_block_ptr = tl.advance(DV_block_ptr, (start_n * BLOCK_M + V_offset, 0))
    offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_N)
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
        offs_m_curr = start_m + offs_m
        q = tl.load(Q_block_ptr)
        if CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= offs_n[None, :], float(
                0.0), float('-inf'))
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i[:, None])
        do = tl.load(DO_block_ptr)
        dv += tl.dot(tl.trans(p), do)
        Di = tl.load(D_ptrs + offs_m_curr)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - Di[:, None]) * sm_scale
        dk += tl.dot(tl.trans(ds), q)
        if not SEQUENCE_PARALLEL:
            dq = tl.load(DQ_block_ptr)
            dq += tl.dot(ds, k)
            tl.store(DQ_block_ptr, dq)
        elif SEQUENCE_PARALLEL:
            if MMA_V3:
                dq = tl.dot(ds, k)
            else:
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds)))
            tl.store(DQ_block_ptr, dq)
        DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
    tl.store(DV_block_ptr, dv)
    tl.store(DK_block_ptr, dk)
