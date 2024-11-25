import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_compute_A(Q, K, GK, A, stride_q1, stride_q2, stride_q3,
    stride_q4, stride_a1, stride_a2, stride_a3, stride_a4, Z, H, N_CTX, D,
    BLOCK_DMODEL_QK: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_k = tl.program_id(2)
    qk_offset = off_hz * stride_q2 + off_k * BLOCK_DMODEL_QK
    a_offset = (off_k * Z * H + off_hz) * stride_a2
    lo = 0
    hi = BLOCK_N
    Q_ptr = Q + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK
        )[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    K_ptr = K + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK
        )[:, None] + tl.arange(0, 16)[None, :] * stride_q4
    GK_K_ptr = GK + qk_offset + start_m * stride_q3 + tl.arange(0,
        BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q4
    GK_Q_ptr = GK + qk_offset + start_m * stride_q3 + tl.arange(0,
        BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    A_ptr = A + a_offset + start_m * stride_a3 + tl.arange(0, 16)[None, :
        ] + tl.arange(0, 16)[:, None] * stride_a4
    for q_high in range(16, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + 
            q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK))
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q = q * q_gk2
        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q4)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q4)
            k_gk = tl.exp(q_normalizer[:, None] - k_gk)
            k = k * k_gk
            qk = tl.dot(q, k, allow_tf32=False)
            tl.store(A_ptr + q_high * stride_a4 + k_high, qk)
    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + 
            q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK))
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q = q * q_gk2
        q_gk3 = tl.exp(q_normalizer[None, :] - q_gk)
        k = tl.load(K_ptr + q_high * stride_q4)
        k = k * tl.trans(q_gk3)
        qk = tl.dot(q, k, allow_tf32=False)
        qk = tl.where(tl.arange(0, 16)[:, None] >= tl.arange(0, 16)[None, :
            ], qk, 0.0)
        tl.store(A_ptr + q_high * stride_a4 + q_high, qk)
