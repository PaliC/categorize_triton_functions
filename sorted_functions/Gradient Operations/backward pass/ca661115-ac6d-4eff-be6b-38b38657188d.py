import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kernel(Q, K, V, sm_scale, Out, DO, DQ, DK, DV, L, D, stride_qz,
    stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn,
    stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, Z, H, N_CTX,
    P_SEQ, num_block_q, num_block_kv, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', CAUSAL: 'tl.constexpr'):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    qk_scale = sm_scale * 1.44269504
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_kz + off_h * stride_kh
    DV += off_z * stride_vz + off_h * stride_vh
    for start_n in range(0, num_block_kv):
        if CAUSAL:
            lo = tl.math.max(start_n * BLOCK_M - P_SEQ, 0)
        else:
            lo = 0
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] *
            stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
            )
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk
            )
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] *
            stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] *
            stride_qk)
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        for start_m in range(lo, num_block_q * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            q = tl.load(q_ptrs)
            if CAUSAL:
                qk = tl.where(P_SEQ + offs_m_curr[:, None] >= offs_n[None,
                    :], float(0.0), float('-inf'))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk *= qk_scale
            l_i = tl.load(l_ptrs + offs_m_curr)
            p = tl.math.exp2(qk - l_i[:, None])
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p), do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            ds = p * dp * sm_scale
            dk += tl.dot(tl.trans(ds), q)
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds, k)
            tl.store(dq_ptrs, dq)
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] *
            stride_kk)
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] *
            stride_qk)
        tl.store(dk_ptrs, dk)
        tl.store(dv_ptrs, dv)
