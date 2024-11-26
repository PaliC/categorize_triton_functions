import triton
import triton.language as tl
import torch

@triton_autotune(configs=_get_bw_configs(), key=['AUTOTUNE_Z', 'H',
    'AUTOTUNE_MAX_SEQ_LEN', 'DimQ', 'DimV', 'BUCKET_FN', 'ATTN_BIAS_TYPE'])
@triton.jit
def _ragged_hstu_attn_bwd(Q, K, V, sort_by_length_indices, seq_offsets, TS,
    TW, PW, Bias, seq2_offsets, num_targets, DOut, DQ, DK, DV, DBias, DTW,
    DPW, LOCK, stride_qm, stride_qh, stride_kn, stride_kh, stride_vn,
    stride_vh, stride_ts, stride_dom, stride_doh, stride_dqm, stride_dqh,
    stride_dkn, stride_dkh, stride_dvn, stride_dvh, alpha, Z, AUTOTUNE_Z, H,
    MAX_SEQ_LEN, AUTOTUNE_MAX_SEQ_LEN, DimQ, DimV, num_buckets, max_pos_ind,
    time_bucket_incr, time_bucket_div, time_delta, CONTEXTUAL_SEQ_LEN:
    'tl.constexpr', MAX_ATTN_LEN: 'tl.constexpr', INVALID_MASK_TYPE:
    'tl.constexpr', CAUSAL: 'tl.constexpr', BUCKET_FN: 'tl.constexpr',
    ATTN_BIAS_TYPE: 'tl.constexpr', USE_TIME_BIAS: 'tl.constexpr',
    USE_POS_BIAS: 'tl.constexpr', FUSED_BIAS_BWD: 'tl.constexpr',
    HAS_MAX_POS_IND: 'tl.constexpr', HAS_MULTIPLE_TARGETS: 'tl.constexpr',
    ALLOW_TF32: 'tl.constexpr', BLOCK_D_Q: 'tl.constexpr', BLOCK_D_V:
    'tl.constexpr', SEQUENCE_PARALLEL: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', UNROLL: 'tl.constexpr',
    HAS_SORT_BY_LENGTH_INDICES: 'tl.constexpr'):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h = off_h
    seq_start = tl.load(seq_offsets + off_z)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = seq_end - seq_start
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z)
    else:
        n_targets = None
    Q = Q + seq_start * stride_qm + off_h * stride_qh
    K = K + seq_start * stride_kn + off_h * stride_kh
    V = V + seq_start * stride_vn + off_h * stride_vh
    DOut = DOut + seq_start * stride_dom + off_h * stride_doh
    DQ = DQ + seq_start * stride_dqm + off_h * stride_dqh
    DK = DK + seq_start * stride_dkn + off_h * stride_dkh
    DV = DV + seq_start * stride_dvn + off_h * stride_dvh
    if ATTN_BIAS_TYPE == 'fused':
        if USE_TIME_BIAS:
            TS = TS + off_z * stride_ts
        if FUSED_BIAS_BWD:
            if USE_TIME_BIAS:
                DTW = DTW + off_hz * (num_buckets + 1)
            if USE_POS_BIAS:
                if HAS_MAX_POS_IND:
                    DPW = DPW + off_hz * (2 * max_pos_ind - 1)
                else:
                    DPW = DPW + off_hz * (2 * MAX_SEQ_LEN - 1)
    elif ATTN_BIAS_TYPE == 'separate':
        seq2_start = tl.load(seq2_offsets + off_z)
        bias_start = seq2_start * H + off_h * seq_len * seq_len
        Bias = Bias + bias_start
        DBias = DBias + bias_start
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len:
            return
        _ragged_hstu_attn_bwd_one_col_block(start_n=start_n, seq_len=
            seq_len, n_targets=n_targets, Q=Q, K=K, V=V, TS=TS, TW=TW, PW=
            PW, Bias=Bias, DOut=DOut, DQ=DQ, DK=DK, DV=DV, DBias=DBias, DTW
            =DTW, DPW=DPW, LOCK=LOCK, stride_qm=stride_qm, stride_kn=
            stride_kn, stride_vn=stride_vn, stride_dom=stride_dom,
            stride_dqm=stride_dqm, stride_dkn=stride_dkn, stride_dvn=
            stride_dvn, alpha=alpha, MAX_SEQ_LEN=MAX_SEQ_LEN, num_buckets=
            num_buckets, max_pos_ind=max_pos_ind, MAX_ATTN_LEN=MAX_ATTN_LEN,
            time_bucket_incr=time_bucket_incr, time_bucket_div=
            time_bucket_div, time_delta=time_delta, INVALID_MASK_TYPE=
            INVALID_MASK_TYPE, CAUSAL=CAUSAL, BUCKET_FN=BUCKET_FN,
            ATTN_BIAS_TYPE=ATTN_BIAS_TYPE, USE_TIME_BIAS=USE_TIME_BIAS,
            USE_POS_BIAS=USE_POS_BIAS, FUSED_BIAS_BWD=FUSED_BIAS_BWD,
            HAS_MAX_POS_IND=HAS_MAX_POS_IND, HAS_MULTIPLE_TARGETS=
            HAS_MULTIPLE_TARGETS, CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
            ALLOW_TF32=ALLOW_TF32, BLOCK_D_Q=BLOCK_D_Q, BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, UNROLL=UNROLL, ATOMIC_ADD=True)
    else:
        for start_n in range(0, seq_len, BLOCK_N):
            _ragged_hstu_attn_bwd_one_col_block(start_n=start_n, seq_len=
                seq_len, n_targets=n_targets, Q=Q, K=K, V=V, TS=TS, TW=TW,
                PW=PW, Bias=Bias, DOut=DOut, DQ=DQ, DK=DK, DV=DV, DBias=
                DBias, DTW=DTW, DPW=DPW, LOCK=LOCK, stride_qm=stride_qm,
                stride_kn=stride_kn, stride_vn=stride_vn, stride_dom=
                stride_dom, stride_dqm=stride_dqm, stride_dkn=stride_dkn,
                stride_dvn=stride_dvn, alpha=alpha, MAX_SEQ_LEN=MAX_SEQ_LEN,
                num_buckets=num_buckets, max_pos_ind=max_pos_ind,
                MAX_ATTN_LEN=MAX_ATTN_LEN, time_bucket_incr=
                time_bucket_incr, time_bucket_div=time_bucket_div,
                time_delta=time_delta, INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL, BUCKET_FN=BUCKET_FN, ATTN_BIAS_TYPE=
                ATTN_BIAS_TYPE, USE_TIME_BIAS=USE_TIME_BIAS, USE_POS_BIAS=
                USE_POS_BIAS, FUSED_BIAS_BWD=FUSED_BIAS_BWD,
                HAS_MAX_POS_IND=HAS_MAX_POS_IND, HAS_MULTIPLE_TARGETS=
                HAS_MULTIPLE_TARGETS, CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32, BLOCK_D_Q=BLOCK_D_Q, BLOCK_D_V=
                BLOCK_D_V, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, UNROLL=UNROLL,
                ATOMIC_ADD=False)
