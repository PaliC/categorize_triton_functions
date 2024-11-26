import triton
import triton.language as tl
import torch

@triton.jit
def bwd_inner_dk_dv(dk, dv, qk_scale, bias_scale, q_ptrs, q_stride, kt, vt,
    B_block_ptr, do_ptrs, do_stride, l_ptrs, D_ptrs, seqlen_q, seqlen_k,
    head_dim, start_k, lo, hi, overflow_size, dropout_p, philox_seed,
    batch_philox_offset, max_seqlen_k, BLOCK_M: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', FULL_BLOCKS:
    'tl.constexpr', CAUSAL: 'tl.constexpr', ENABLE_DROPOUT: 'tl.constexpr',
    PADDED_HEAD: 'tl.constexpr', BIAS_TYPE: 'tl.constexpr'):
    offs_k = start_k + tl.arange(0, BLOCK_N)
    offs_q = tl.arange(0, BLOCK_M)
    ld_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
    q_ptrs += lo * q_stride
    do_ptrs += lo * do_stride
    if BIAS_TYPE == 1:
        B_block_ptr = tl.advance(B_block_ptr, (lo, 0))
    """
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)

    dV = (QK)^T dO

    dV1 = qk11 dO1 + qk21 dO2 = q1 k1 dO1 + q2 k1 dO2
    dV2 = qk12 dO1 + qk22 dO2 = q1 k2 dO1 + q2 k2 dO2
                                ~~~~~ = 0
    start_k: select k and dV
    start_q: select q and dO
    """
    for start_q in range(lo, hi, BLOCK_M):
        offs_q_curr = offs_q[:, None] + start_q
        if not FULL_BLOCKS:
            q = load_fn(q_ptrs, offs_q + start_q, ld_offs_d, seqlen_q, head_dim
                )
        else:
            q = load_fn(q_ptrs, None, ld_offs_d, seqlen_q, head_dim)
        if not FULL_BLOCKS:
            do = load_fn(do_ptrs, offs_q + start_q, ld_offs_d, seqlen_q,
                head_dim)
        else:
            do = load_fn(do_ptrs, None, ld_offs_d, seqlen_q, head_dim)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if not FULL_BLOCKS:
            if overflow_size > 0:
                boundary_n = tl.full((BLOCK_N,), seqlen_q, dtype=tl.int32)
                mask = offs_q_curr < boundary_n[None, :]
                qk = tl.where(mask, qk, float('-inf'))
        if CAUSAL:
            qk = tl.where(offs_q_curr >= offs_k[None, :], qk, float('-inf'))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            bias = tl.load(B_block_ptr, boundary_check=(0, 1),
                padding_option='zero')
            qk += bias * bias_scale
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
        qk += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, q, kt)
        if FULL_BLOCKS:
            Di = tl.load(D_ptrs + offs_q_curr)
            l_i = tl.load(l_ptrs + offs_q_curr)
        else:
            boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=
                tl.int32)
            d_lse_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
            d_lse_padding = tl.full((BLOCK_M,), 0, dtype=tl.float32)
            Di = tl.load(D_ptrs + offs_q_curr, mask=d_lse_ptrs_mask[:, None
                ], other=d_lse_padding[:, None])
            l_i = tl.load(l_ptrs + offs_q_curr, mask=d_lse_ptrs_mask[:,
                None], other=d_lse_padding[:, None])
        p = tl.math.exp2(qk_scale * qk - l_i)
        if not FULL_BLOCKS or CAUSAL:
            if qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)
        if ENABLE_DROPOUT:
            philox_offset = (batch_philox_offset + start_q * max_seqlen_k +
                start_k)
            keep = dropout_mask(philox_seed, philox_offset, dropout_p,
                BLOCK_M, BLOCK_N, max_seqlen_k)
            if BLOCK_M == 1:
                dv += tl.where(keep, p / (1 - dropout_p), 0.0) * do
            else:
                dv += tl.dot(tl.trans(tl.where(keep, p / (1 - dropout_p), 
                    0.0)), do)
        elif BLOCK_M == 1:
            dv += p * do
        else:
            dv += tl.dot(tl.trans(p), do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, vt)
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        ds = p * (dp - Di)
        if BLOCK_M == 1:
            dk += ds * q
        else:
            dk += tl.dot(tl.trans(ds), q)
        q_ptrs += q_stride * BLOCK_M
        do_ptrs += do_stride * BLOCK_M
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (BLOCK_M, 0))
    return dk, dv
