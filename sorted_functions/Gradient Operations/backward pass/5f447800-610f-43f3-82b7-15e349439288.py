import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_blocked_kernel_one_col(start_n, Q, K, V, Q_idx, K_idx, DO, DQ, DK,
    DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_dom,
    stride_dqm, stride_dkn, stride_dvn, stride_q_idxm, stride_k_idxn,
    seqlen_q, block_size, headdim, v_headdim, smooth_block, BLOCK_HEADDIM:
    'tl.constexpr', V_BLOCK_HEADDIM: 'tl.constexpr', EVEN_HEADDIM:
    'tl.constexpr', EVEN_V_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr'):
    block_id = start_n // block_size
    block_offs = seqlen_q + start_n % block_size * BLOCK_M - (block_size - 1
        ) * BLOCK_M // 2
    begin_m = block_id * BLOCK_M * block_size
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_vd = tl.arange(0, V_BLOCK_HEADDIM)
    k_idx_ptrs = K_idx + offs_n * stride_k_idxn
    k_idx = tl.load(k_idx_ptrs)
    k_ptrs = K + (k_idx[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (k_idx[:, None] * stride_vn + offs_vd[None, :])
    dv = tl.zeros([BLOCK_N, V_BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if EVEN_HEADDIM:
        k = tl.load(k_ptrs)
    else:
        k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    if EVEN_V_HEADDIM:
        v = tl.load(v_ptrs)
    else:
        v = tl.load(v_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)
    end_m = tl.minimum((block_id + 1) * BLOCK_M * block_size, seqlen_q)
    for start_m in range(begin_m, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        if smooth_block:
            q_idx_ptrs = (start_m + block_offs + offs_m
                ) * stride_q_idxm % seqlen_q
        else:
            q_idx_ptrs = (start_m + offs_m) * stride_q_idxm
        q_idx = tl.load(Q_idx + q_idx_ptrs)
        q_ptrs = Q + (q_idx[:, None] * stride_qm + offs_d[None, :])
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        lse_i = tl.load(LSE + q_idx)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])
        do_ptrs = DO + (q_idx[:, None] * stride_dom + offs_vd[None, :])
        if EVEN_V_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=offs_vd[None, :] < v_headdim, other=0.0)
        dv += tl.dot(tl.trans(p), do)
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v))
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        Di = tl.load(D + q_idx)
        ds = p * (dp - Di[:, None]) * softmax_scale
        dk += tl.dot(tl.trans(ds), q)
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        dq_ptrs = DQ + (q_idx[:, None] * stride_dqm + offs_d[None, :])
        dq = tl.dot(ds, k)
        if EVEN_HEADDIM:
            tl.atomic_add(dq_ptrs, dq)
        else:
            tl.atomic_add(dq_ptrs, dq, mask=offs_d[None, :] < headdim)
    dv_ptrs = DV + (k_idx[:, None] * stride_dvn + offs_vd[None, :])
    dk_ptrs = DK + (k_idx[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dx(dk_ptrs, dk, offs_d, headdim, even_headdim=EVEN_HEADDIM)
    _bwd_store_dx(dv_ptrs, dv, offs_vd, v_headdim, even_headdim=EVEN_V_HEADDIM)
