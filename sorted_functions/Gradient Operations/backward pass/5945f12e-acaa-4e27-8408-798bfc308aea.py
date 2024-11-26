import triton
import triton.language as tl
import torch

@triton.jit
def bwd_inner_dq(dq, qk_scale, bias_scale, DB_block_ptr, store_db, q,
    kt_ptrs, k_stride, vt_ptrs, v_stride, B_block_ptr, do, Di, l_i,
    seqlen_q, seqlen_k, head_dim, start_q, lo, hi, dropout_p, philox_seed,
    batch_philox_offset, max_seqlen_k, BLOCK_M: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', FULL_BLOCKS:
    'tl.constexpr', CAUSAL: 'tl.constexpr', ENABLE_DROPOUT: 'tl.constexpr',
    PADDED_HEAD: 'tl.constexpr', BIAS_TYPE: 'tl.constexpr'):
    offs_q = start_q + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    ld_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
    kt_ptrs += lo * k_stride
    vt_ptrs += lo * v_stride
    if BIAS_TYPE == 1:
        B_block_ptr = tl.advance(B_block_ptr, (0, lo))
        DB_block_ptr = tl.advance(DB_block_ptr, (0, lo))
    """
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)
    """
    for start_k in range(lo, hi, BLOCK_N):
        offs_k_curr = offs_k[None, :] + start_k
        if not FULL_BLOCKS:
            kt = load_fn(kt_ptrs, ld_offs_d, offs_k + start_k, head_dim,
                seqlen_k)
        else:
            kt = load_fn(kt_ptrs, ld_offs_d, None, head_dim, seqlen_k)
        if not FULL_BLOCKS:
            vt = load_fn(vt_ptrs, ld_offs_d, offs_k + start_k, head_dim,
                seqlen_k)
        else:
            vt = load_fn(vt_ptrs, ld_offs_d, None, head_dim, seqlen_k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, q, kt)
        if not FULL_BLOCKS:
            k_boundary = tl.full((BLOCK_M,), seqlen_k, dtype=tl.int32)
            mask = offs_k_curr < k_boundary[:, None]
            qk = tl.where(mask, qk, float('-inf'))
        if CAUSAL:
            qk = tl.where(offs_q[:, None] >= offs_k_curr, qk, float('-inf'))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            bias = tl.load(B_block_ptr, boundary_check=(0, 1),
                padding_option='zero')
            qk += bias * bias_scale
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
        p = tl.math.exp2(qk_scale * qk - l_i[:, None])
        if not FULL_BLOCKS or CAUSAL:
            if qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, do, vt)
        if ENABLE_DROPOUT:
            philox_offset = (batch_philox_offset + start_q * max_seqlen_k +
                start_k)
            keep = dropout_mask(philox_seed, philox_offset, dropout_p,
                BLOCK_M, BLOCK_N, max_seqlen_k)
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        ds = p * (dp - Di[:, None])
        if BLOCK_M == 1:
            dq += tl.view(kt, [BLOCK_DMODEL]) * ds
        else:
            dq = tl.dot(ds, tl.trans(kt), acc=dq)
        if BIAS_TYPE == 1:
            if store_db:
                tl.store(DB_block_ptr, ds, boundary_check=(0, 1))
        kt_ptrs += BLOCK_N * k_stride
        vt_ptrs += BLOCK_N * v_stride
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))
            DB_block_ptr = tl.advance(DB_block_ptr, (0, BLOCK_N))
    return dq
