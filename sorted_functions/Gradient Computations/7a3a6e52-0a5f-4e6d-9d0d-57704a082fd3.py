import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kernel_dqk(Q, K, GK, DA, DQ, DK, DGK, stride_q1, stride_q2,
    stride_q3, stride_q4, stride_a1, stride_a2, stride_a3, stride_a4, Z, H,
    N_CTX, D, BLOCK_DMODEL_QK: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_k = tl.program_id(2)
    qk_offset = off_hz * stride_q2 + BLOCK_DMODEL_QK * off_k
    a_offset = off_hz * stride_a2
    lo = 0
    hi = BLOCK_N
    Q_ptr = Q + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK
        )[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    DQ_ptr = DQ + qk_offset + start_m * stride_q3 + tl.arange(0,
        BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    K_ptr = K + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK
        )[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    GK_K_ptr = GK + qk_offset + start_m * stride_q3 + tl.arange(0,
        BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    GK_Q_ptr = GK + qk_offset + start_m * stride_q3 + tl.arange(0,
        BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    DA_ptr = DA + a_offset + start_m * stride_a3 + tl.arange(0, 16)[None, :
        ] + tl.arange(0, 16)[:, None] * stride_a4
    for q_high in range(lo + 16, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + 
            q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK))
        dq2 = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q4)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q4)
            dqk = tl.load(DA_ptr + q_high * stride_a4 + k_high)
            k_gk = tl.exp(q_normalizer[None, :] - k_gk)
            k = k * k_gk
            dq2 += tl.dot(dqk, k, allow_tf32=False)
        dq2 = dq2
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4)
        q_gk = tl.exp(q_gk - q_normalizer[None, :])
        dq = dq2 * q_gk
        dq_gk = dq * q
        DQ_ptr = DQ + qk_offset + start_m * stride_q3 + tl.arange(0,
            BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None
            ] * stride_q4 + q_high * stride_q4
        tl.store(DQ_ptr, dq)
        DGK_Q_ptr = DGK + qk_offset + start_m * stride_q3 + tl.arange(0,
            BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None
            ] * stride_q4 + q_high * stride_q4
        tl.store(DGK_Q_ptr, dq_gk)
    tl.debug_barrier()
    for k_high in range(lo, hi - 16, 16):
        k = tl.load(K_ptr + k_high * stride_q4)
        k_gk = tl.load(GK_K_ptr + k_high * stride_q4)
        dk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        dgk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        for q_high in range(k_high + 16, hi, 16):
            q = tl.load(Q_ptr + q_high * stride_q4)
            q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + 
                q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK))
            q_gk = tl.load(GK_Q_ptr + q_high * stride_q4)
            q_gk = tl.exp(q_gk - q_normalizer[None, :])
            q = q * q_gk
            dqk = tl.load(DA_ptr + q_high * stride_a4 + k_high)
            k_gk2 = tl.exp(q_normalizer[None, :] - k_gk)
            dk2 = tl.dot(tl.trans(dqk), q, allow_tf32=False)
            dk += dk2 * k_gk2
            dgk -= dk2 * k * k_gk2
        DK_ptr = DK + qk_offset + start_m * stride_q3 + tl.arange(0,
            BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None
            ] * stride_q4 + k_high * stride_q4
        tl.store(DK_ptr, dk)
        DGK_K_ptr = DGK + qk_offset + start_m * stride_q3 + tl.arange(0,
            BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None
            ] * stride_q4 + k_high * stride_q4
        prev = tl.load(DGK_K_ptr)
        tl.store(DGK_K_ptr, prev + dgk)
    tl.debug_barrier()
    DK_ptr = DK + qk_offset + start_m * stride_q3 + tl.arange(0,
        BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    DGK_K_ptr = DGK + qk_offset + start_m * stride_q3 + tl.arange(0,
        BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    DQ_ptr = DQ + qk_offset + start_m * stride_q3 + tl.arange(0,
        BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + 
            q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK))
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q2 = q * q_gk2
        q_gk3 = tl.exp(q_normalizer[None, :] - q_gk)
        k = tl.load(K_ptr + q_high * stride_q4)
        k2 = k * q_gk3
        dqk = tl.load(DA_ptr + q_high * stride_a4 + q_high)
        dqk = tl.where(tl.arange(0, 16)[:, None] >= tl.arange(0, 16)[None,
            :], dqk, 0.0)
        dk2 = tl.dot(tl.trans(dqk), q2, allow_tf32=False)
        dk = dk2 * q_gk3
        prev_dk = tl.load(DK_ptr + q_high * stride_q4)
        tl.store(DK_ptr + q_high * stride_q4, dk + prev_dk)
        dgk = -dk * k
        dq2 = tl.dot(dqk, k2, allow_tf32=False)
        dq = dq2 * q_gk2
        prev_dq = tl.load(DQ_ptr + q_high * stride_q4)
        tl.store(DQ_ptr + q_high * stride_q4, dq + prev_dq)
        dgk += dq * q
        prev_dq_gk = tl.load(DGK_K_ptr + q_high * stride_q4)
        tl.store(DGK_K_ptr + q_high * stride_q4, dgk + prev_dq_gk)
