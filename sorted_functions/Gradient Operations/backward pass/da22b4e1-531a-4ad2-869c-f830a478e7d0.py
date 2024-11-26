import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kernel(Q, K, V, sm_scale, Out, DO, DQ, DK, DV, stride_qz,
    stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn,
    stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, Z, H, N_CTX,
    num_block, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr'):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_qz + off_h * stride_qh
    V += off_z * stride_qz + off_h * stride_qh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
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
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        v = v * sm_scale
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            q = tl.load(q_ptrs)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k), allow_tf32=False)
            p = qk
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p), do)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            dp += tl.dot(do, tl.trans(v))
            ds = dp
            dk += tl.dot(tl.trans(ds), q)
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds, k)
            tl.store(dq_ptrs, dq)
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        dv *= sm_scale
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] *
            stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] *
            stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)
