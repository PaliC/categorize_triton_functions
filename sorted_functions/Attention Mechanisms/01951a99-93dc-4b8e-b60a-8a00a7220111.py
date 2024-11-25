import triton
import triton.language as tl
import torch

@triton.jit
def _total_attention_kernel(Q, K, L, TA, sm_scale, stride_qz, stride_qh,
    stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, Z, H,
    M, N, P_SEQ, num_groups, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', CAUSAL: 'tl.constexpr',
    DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N: 'tl.constexpr'):
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: 'tl.constexpr' = 1.4426950408889634
    qk_scale = sm_scale * log2e
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    L += (off_z * H + off_h) * M
    TA += (off_z * H + off_h) * N
    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = lo // BLOCK_M * BLOCK_M
    else:
        lo = 0
    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] *
        stride_qk)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    ta_ptrs = TA + offs_n
    if DIVISIBLE_N:
        k = tl.load(k_ptrs)
    else:
        mask_n = offs_n < N
        k = tl.load(k_ptrs, mask=mask_n[:, None])
    tot_attn = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
        else:
            mask_m = offs_m < M
            valid_mask = mask_m[:, None]
            q = tl.load(q_ptrs, mask=mask_m[:, None])
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))
        if DIVISIBLE_M:
            l = tl.load(L + offs_m)
        else:
            l = tl.load(L + offs_m, mask=mask_m)
        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
        if not DIVISIBLE_M:
            p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)
        tot_attn += tl.sum(p, 0)
        q_ptrs += BLOCK_M * stride_qm
    if DIVISIBLE_N:
        tl.store(ta_ptrs, tot_attn)
    else:
        tl.store(ta_ptrs, tot_attn, mask=mask_n)
