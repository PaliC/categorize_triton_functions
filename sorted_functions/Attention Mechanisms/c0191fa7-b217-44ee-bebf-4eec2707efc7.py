import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kernel(Q, K, V, sm_scale, Out, DO, DQ, DK, DV, L, M, D, sqz, sqh,
    sqm, sqd, skz, skh, skn, skd, svz, svh, svn, svd, Z, H, N_CTX_Q,
    N_CTX_KV, BLOCK: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr',
    N_PREFIX_Q: 'tl.constexpr'):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    BLOCK_M: 'tl.constexpr' = BLOCK
    BLOCK_N: 'tl.constexpr' = BLOCK
    Q += off_z * sqz + off_h * sqh
    K += off_z * skz + off_h * skh
    V += off_z * svz + off_h * svh
    DO += off_z * sqz + off_h * sqh
    DQ += off_z * sqz + off_h * sqh
    DK += off_z * skz + off_h * skh
    DV += off_z * svz + off_h * svh
    offs_d = tl.arange(0, BLOCK_DMODEL)
    D_ptrs = D + off_hz * N_CTX_Q
    m_ptrs = M + off_hz * N_CTX_Q
    for start_n in range(0, N_CTX_KV, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        v_ptrs = V + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        if start_n < N_PREFIX_Q * BLOCK_M:
            start_q_index = 0
        elif N_CTX_Q <= start_n - N_PREFIX_Q * BLOCK_M:
            start_q_index = start_n - N_PREFIX_Q * BLOCK_M
        else:
            first_start_m = start_n - N_PREFIX_Q * BLOCK_M
            first_start_m = tl.multiple_of(first_start_m, BLOCK_M)
            offs_m = first_start_m + tl.arange(0, BLOCK_M)
            offs_m_real = offs_m + N_PREFIX_Q * BLOCK_M
            offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0
                )
            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            dq_ptrs = DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk,
                float('-inf'))
            m = tl.load(m_ptrs + offs_m)
            m_ = m
            last_p = tl.exp(qk * sm_scale - m_[:, None])
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(last_p), do, allow_tf32=False)
            Di = tl.load(D_ptrs + offs_m)
            last_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:,
                None]
            last_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            ds = last_p * last_dp * sm_scale
            dk += tl.dot(tl.trans(ds), q, allow_tf32=False)
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds, k, allow_tf32=False)
            tl.store(dq_ptrs, dq)
            start_q_index = first_start_m + BLOCK_M
        for start_m in range(start_q_index, N_CTX_Q, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + tl.arange(0, BLOCK_M)
            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            dq_ptrs = DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale
            landmark_qk = tl.max(tl.where(tl.arange(0, BLOCK_N)[None, :] ==
                BLOCK_N - 1, qk, float('-inf')), 1)
            normal_qk = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N -
                1, float('-inf'), qk)
            m = tl.load(m_ptrs + offs_m)
            m_ = m
            p = tl.exp(landmark_qk - m_)
            do = tl.load(do_ptrs)
            normal_m = tl.max(normal_qk, 1)
            normal_p = tl.exp(normal_qk - normal_m[:, None])
            normal_p_normalized = normal_p / tl.sum(normal_p, 1)[:, None]
            normal_kv = tl.dot(normal_p_normalized, v, allow_tf32=False)
            normal_D = tl.sum(do * normal_kv, 1)
            dv += tl.dot(tl.trans(p[:, None] * normal_p_normalized), do,
                allow_tf32=False)
            Di = tl.load(D_ptrs + offs_m)
            dp = tl.zeros([BLOCK_M], dtype=tl.float32) - Di
            dp += normal_D
            landmark_ds = p * dp
            normal_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32
                ) - normal_D[:, None]
            normal_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            normal_ds = p[:, None] * normal_p_normalized * normal_dp
            ds = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1,
                landmark_ds[:, None], normal_ds)
            ds *= sm_scale
            dk += tl.dot(tl.trans(ds), q, allow_tf32=False)
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds, k, allow_tf32=False)
            tl.store(dq_ptrs, dq)
        dv_ptrs = DV + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dk_ptrs = DK + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)
