import triton
import triton.language as tl
import torch

@triton.jit
def _ragged_hstu_attn_bwd_one_col_block(start_n, seq_len, n_targets, Q, K,
    V, TS, TW, PW, Bias, DOut, DQ, DK, DV, DBias, DTW, DPW, LOCK, stride_qm,
    stride_kn, stride_vn, stride_dom, stride_dqm, stride_dkn, stride_dvn,
    alpha, MAX_SEQ_LEN, num_buckets, max_pos_ind, time_bucket_incr,
    time_bucket_div, time_delta, MAX_ATTN_LEN: 'tl.constexpr',
    INVALID_MASK_TYPE: 'tl.constexpr', CAUSAL: 'tl.constexpr', BUCKET_FN:
    'tl.constexpr', ATTN_BIAS_TYPE: 'tl.constexpr', USE_TIME_BIAS:
    'tl.constexpr', USE_POS_BIAS: 'tl.constexpr', FUSED_BIAS_BWD:
    'tl.constexpr', HAS_MAX_POS_IND: 'tl.constexpr', HAS_MULTIPLE_TARGETS:
    'tl.constexpr', CONTEXTUAL_SEQ_LEN: 'tl.constexpr', ALLOW_TF32:
    'tl.constexpr', BLOCK_D_Q: 'tl.constexpr', BLOCK_D_V: 'tl.constexpr',
    BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', UNROLL:
    'tl.constexpr', ATOMIC_ADD: 'tl.constexpr'):
    if INVALID_MASK_TYPE == 'lower_triangular':
        if HAS_MULTIPLE_TARGETS:
            low = start_n
            if MAX_ATTN_LEN > 0:
                high = start_n + MAX_ATTN_LEN + BLOCK_N
                high = high if high + n_targets < seq_len else seq_len
            else:
                high = seq_len
        else:
            low = start_n
            if MAX_ATTN_LEN > 0:
                high = start_n + MAX_ATTN_LEN + BLOCK_N
                high = high if high < seq_len else seq_len
            else:
                high = seq_len
        if CONTEXTUAL_SEQ_LEN > 0:
            contextual_block_end = tl.cdiv(CONTEXTUAL_SEQ_LEN, BLOCK_M
                ) * BLOCK_M
            if low < contextual_block_end:
                low = contextual_block_end
    elif INVALID_MASK_TYPE == 'upper_triangular':
        low = 0
        high = start_n + BLOCK_N
    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    q_ptrs_trans = Q + (offs_m[None, :] * stride_qm + offs_qk_d[:, None])
    dq_ptrs_trans = DQ + (offs_m[None, :] * stride_dqm + offs_qk_d[:, None])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
    mask_n = offs_n < seq_len
    ts_0_ptrs = None
    ts_1_ptrs = None
    ts_1 = None
    off_bias_trans = None
    bias_ptrs_trans = None
    dbias_ptrs_trans = None
    if ATTN_BIAS_TYPE == 'fused' and USE_TIME_BIAS:
        ts_0_ptrs = TS + offs_m
        ts_1_ptrs = TS + offs_n
        if CAUSAL:
            ts_1 = tl.load(ts_1_ptrs, mask=mask_n)
        else:
            ts_1 = tl.load(ts_1_ptrs + 1, mask=mask_n)
    elif ATTN_BIAS_TYPE == 'separate':
        off_bias_trans = offs_m[None, :] * seq_len + offs_n[:, None]
        bias_ptrs_trans = Bias + off_bias_trans
        dbias_ptrs_trans = DBias + off_bias_trans
    do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    if HAS_MULTIPLE_TARGETS:
        if INVALID_MASK_TYPE == 'lower_triangular':
            pos_offs_n = tl.where(offs_n < seq_len - n_targets, offs_n, 
                seq_len - n_targets)
        elif INVALID_MASK_TYPE == 'upper_triangular':
            pos_offs_n = tl.where(offs_n > n_targets - 1, offs_n, n_targets - 1
                )
    else:
        pos_offs_n = offs_n
    if CONTEXTUAL_SEQ_LEN > 0 and INVALID_MASK_TYPE == 'lower_triangular':
        for start_m in range(0, CONTEXTUAL_SEQ_LEN, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _ragged_hstu_attn_bwd_one_block(start_m=start_m,
                offs_n=offs_n, offs_m=offs_m, q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans, mask_n=mask_n, ts_0_ptrs=
                ts_0_ptrs, ts_1=ts_1, bias_ptrs_trans=bias_ptrs_trans,
                dbias_ptrs_trans=dbias_ptrs_trans, do_ptrs=do_ptrs, dk=dk,
                dv=dv, k=k, v=v, pos_offs_n=pos_offs_n, seq_len=seq_len,
                n_targets=n_targets, TW=TW, PW=PW, DTW=DTW, DPW=DPW, LOCK=
                LOCK, stride_qm=stride_qm, stride_dom=stride_dom,
                stride_dqm=stride_dqm, alpha=alpha, MAX_SEQ_LEN=MAX_SEQ_LEN,
                num_buckets=num_buckets, max_pos_ind=max_pos_ind,
                MAX_ATTN_LEN=MAX_ATTN_LEN, time_bucket_incr=
                time_bucket_incr, time_bucket_div=time_bucket_div,
                time_delta=time_delta, INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL, BUCKET_FN=BUCKET_FN, ATTN_BIAS_TYPE=
                ATTN_BIAS_TYPE, USE_TIME_BIAS=USE_TIME_BIAS, USE_POS_BIAS=
                USE_POS_BIAS, FUSED_BIAS_BWD=FUSED_BIAS_BWD,
                HAS_MAX_POS_IND=HAS_MAX_POS_IND, HAS_MULTIPLE_TARGETS=
                HAS_MULTIPLE_TARGETS, CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                ATOMIC_ADD=ATOMIC_ADD)
    for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _ragged_hstu_attn_bwd_one_block(start_m=start_m, offs_n=
            offs_n, offs_m=offs_m, q_ptrs_trans=q_ptrs_trans, dq_ptrs_trans
            =dq_ptrs_trans, mask_n=mask_n, ts_0_ptrs=ts_0_ptrs, ts_1=ts_1,
            bias_ptrs_trans=bias_ptrs_trans, dbias_ptrs_trans=
            dbias_ptrs_trans, do_ptrs=do_ptrs, dk=dk, dv=dv, k=k, v=v,
            pos_offs_n=pos_offs_n, seq_len=seq_len, n_targets=n_targets, TW
            =TW, PW=PW, DTW=DTW, DPW=DPW, LOCK=LOCK, stride_qm=stride_qm,
            stride_dom=stride_dom, stride_dqm=stride_dqm, alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN, num_buckets=num_buckets, max_pos_ind=
            max_pos_ind, MAX_ATTN_LEN=MAX_ATTN_LEN, time_bucket_incr=
            time_bucket_incr, time_bucket_div=time_bucket_div, time_delta=
            time_delta, INVALID_MASK_TYPE=INVALID_MASK_TYPE, CAUSAL=CAUSAL,
            BUCKET_FN=BUCKET_FN, ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
            USE_TIME_BIAS=USE_TIME_BIAS, USE_POS_BIAS=USE_POS_BIAS,
            FUSED_BIAS_BWD=FUSED_BIAS_BWD, HAS_MAX_POS_IND=HAS_MAX_POS_IND,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS, CONTEXTUAL_SEQ_LEN=
            CONTEXTUAL_SEQ_LEN, ALLOW_TF32=ALLOW_TF32, BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N, ATOMIC_ADD=ATOMIC_ADD)
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
    dk = dk * alpha
    tl.store(dv_ptrs, dv, mask=mask_n[:, None])
    tl.store(dk_ptrs, dk, mask=mask_n[:, None])
