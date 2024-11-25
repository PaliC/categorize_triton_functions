import triton
import triton.language as tl
import torch

@triton.jit
def _ragged_hstu_attn_fwd_compute(Q, K, V, seq_offsets, TS, TW, PW, Bias,
    seq2_offsets, delta_x_offsets, num_targets, Scale, Out, stride_qm,
    stride_qh, stride_kn, stride_kh, stride_vn, stride_vh, stride_sz,
    stride_sm, stride_ts, stride_om, stride_oh, alpha, Z, H, MAX_SEQ_LEN,
    DimQ, DimV, DeltaSize, num_buckets, max_pos_ind, time_bucket_incr,
    time_bucket_div, time_delta, contextual_seq_len, off_z, off_h, pid,
    INVALID_MASK_TYPE: 'tl.constexpr', CAUSAL: 'tl.constexpr', BUCKET_FN:
    'tl.constexpr', ATTN_BIAS_TYPE: 'tl.constexpr', USE_TIME_BIAS:
    'tl.constexpr', USE_POS_BIAS: 'tl.constexpr', HAS_MAX_POS_IND:
    'tl.constexpr', HAS_MULTIPLE_TARGETS: 'tl.constexpr', HAS_ATTN_SCALE:
    'tl.constexpr', IS_DELTA_Q: 'tl.constexpr', ALLOW_TF32: 'tl.constexpr',
    BLOCK_D_Q: 'tl.constexpr', BLOCK_D_V: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', max_attn_len: 'tl.constexpr',
    HAS_MAX_ATTN_LEN: 'tl.constexpr', HAS_CONTEXTUAL_SEQ_LEN: 'tl.constexpr'):
    seq_start = tl.load(seq_offsets + off_z)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = seq_end - seq_start
    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        delta_start = tl.load(delta_x_offsets + off_z * DeltaSize)
        start_m = start_m_delta + delta_start - seq_start
    else:
        start_m_delta = 0
        start_m = pid * BLOCK_M
    if start_m < seq_len:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z)
        else:
            n_targets = None
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        if IS_DELTA_Q:
            Q_block_ptr = tl.make_block_ptr(base=Q + off_h * stride_qh + 
                off_z * DeltaSize * stride_qm, shape=(DeltaSize, BLOCK_D_Q),
                strides=(stride_qm, 1), offsets=(start_m_delta, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q), order=(1, 0))
        else:
            Q_block_ptr = tl.make_block_ptr(base=Q + off_h * stride_qh + 
                seq_start * stride_qm, shape=(seq_len, BLOCK_D_Q), strides=
                (stride_qm, 1), offsets=(start_m, 0), block_shape=(BLOCK_M,
                BLOCK_D_Q), order=(1, 0))
        K_block_ptr = tl.make_block_ptr(base=K + off_h * stride_kh + 
            seq_start * stride_kn, shape=(BLOCK_D_Q, seq_len), strides=(1,
            stride_kn), offsets=(0, 0), block_shape=(BLOCK_D_Q, BLOCK_N),
            order=(0, 1))
        V_block_ptr = tl.make_block_ptr(base=V + off_h * stride_vh + 
            seq_start * stride_vn, shape=(seq_len, BLOCK_D_V), strides=(
            stride_vn, 1), offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_D_V),
            order=(1, 0))
        mask_m = offs_m < seq_len
        if ATTN_BIAS_TYPE == 'fused' and USE_TIME_BIAS:
            ts_0_ptrs = TS + off_z * stride_ts + offs_m
            ts_1_ptrs = TS + off_z * stride_ts + offs_n
            if CAUSAL:
                ts_0 = tl.load(ts_0_ptrs + 1, mask=mask_m)
            else:
                ts_0 = tl.load(ts_0_ptrs, mask=mask_m)
        elif ATTN_BIAS_TYPE == 'separate':
            seq2_start = tl.load(seq2_offsets + off_z)
            bias_start = seq2_start * H + off_h * seq_len * seq_len
            off_bias = offs_m[:, None] * seq_len + offs_n[None, :]
            bias_ptrs = Bias + bias_start + off_bias
        if HAS_ATTN_SCALE:
            scale_ptrs = Scale + off_z * stride_sz
            attn_scale = tl.load(scale_ptrs + offs_m * stride_sm, mask=
                offs_m < seq_len)
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option='zero')
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if INVALID_MASK_TYPE == 'lower_triangular':
            if HAS_MULTIPLE_TARGETS:
                if HAS_MAX_ATTN_LEN:
                    start_m_index = (seq_len - n_targets if start_m > 
                        seq_len - n_targets else start_m)
                    low = start_m_index - max_attn_len
                    low = low if low > 0 else 0
                else:
                    low = 0
                uih_end = (seq_len - n_targets + BLOCK_N - 1
                    ) // BLOCK_N * BLOCK_N
                if uih_end < start_m:
                    high = seq_len - n_targets
                else:
                    high = start_m + BLOCK_M
                if HAS_CONTEXTUAL_SEQ_LEN:
                    if start_m < contextual_seq_len:
                        high = seq_len - n_targets
            else:
                if HAS_MAX_ATTN_LEN:
                    low = start_m - max_attn_len
                    low = low if low > 0 else 0
                else:
                    low = 0
                high = start_m + BLOCK_M
                if HAS_CONTEXTUAL_SEQ_LEN:
                    if start_m < contextual_seq_len:
                        high = seq_len
        elif INVALID_MASK_TYPE == 'upper_triangular':
            low = start_m
            high = seq_len
        if low > 0:
            K_block_ptr = tl.advance(K_block_ptr, (0, low))
            V_block_ptr = tl.advance(V_block_ptr, (low, 0))
        for start_n in range(low, high, BLOCK_N):
            cur_offs_n = offs_n + start_n
            mask_n = cur_offs_n < seq_len
            acc += _ragged_hstu_attn_fwd_one_block(start_n=start_n, seq_len
                =seq_len, offs_m=offs_m, offs_n=cur_offs_n, mask_m=mask_m,
                mask_n=mask_n, q=q, K_block_ptr=K_block_ptr, V_block_ptr=
                V_block_ptr, n_targets=n_targets if HAS_MULTIPLE_TARGETS else
                None, ts_1_ptrs=ts_1_ptrs if ATTN_BIAS_TYPE == 'fused' and
                USE_TIME_BIAS else None, ts_0=ts_0 if ATTN_BIAS_TYPE ==
                'fused' and USE_TIME_BIAS else None, TW=TW, PW=PW, alpha=
                alpha, MAX_SEQ_LEN=MAX_SEQ_LEN, num_buckets=num_buckets,
                max_pos_ind=max_pos_ind, max_attn_len=max_attn_len,
                time_bucket_incr=time_bucket_incr, time_bucket_div=
                time_bucket_div, time_delta=time_delta, bias_ptrs=bias_ptrs if
                ATTN_BIAS_TYPE == 'separate' else None, attn_scale=
                attn_scale if HAS_ATTN_SCALE else None, contextual_seq_len=
                contextual_seq_len, INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL, BUCKET_FN=BUCKET_FN, ATTN_BIAS_TYPE=
                ATTN_BIAS_TYPE, USE_TIME_BIAS=USE_TIME_BIAS, USE_POS_BIAS=
                USE_POS_BIAS, HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS, HAS_ATTN_SCALE=
                HAS_ATTN_SCALE, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN, IS_DELTA_Q=
                IS_DELTA_Q, ALLOW_TF32=ALLOW_TF32, BLOCK_M=BLOCK_M, BLOCK_N
                =BLOCK_N)
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if HAS_MULTIPLE_TARGETS and INVALID_MASK_TYPE == 'lower_triangular':
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                offset = low_delta - uih_end
                K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
                for start_delta in tl.range(low_delta, high_delta, BLOCK_N,
                    num_stages=0):
                    cur_offs_n = offs_n + start_delta
                    mask_n = cur_offs_n < seq_len
                    acc += _ragged_hstu_attn_fwd_one_block(start_n=
                        start_delta, seq_len=seq_len, offs_m=offs_m, offs_n
                        =cur_offs_n, mask_m=mask_m, mask_n=mask_n, q=q,
                        K_block_ptr=K_block_ptr, V_block_ptr=V_block_ptr,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else
                        None, ts_1_ptrs=ts_1_ptrs if ATTN_BIAS_TYPE ==
                        'fused' and USE_TIME_BIAS else None, ts_0=ts_0 if 
                        ATTN_BIAS_TYPE == 'fused' and USE_TIME_BIAS else
                        None, TW=TW, PW=PW, alpha=alpha, MAX_SEQ_LEN=
                        MAX_SEQ_LEN, num_buckets=num_buckets, max_pos_ind=
                        max_pos_ind, max_attn_len=max_attn_len,
                        time_bucket_incr=time_bucket_incr, time_bucket_div=
                        time_bucket_div, time_delta=time_delta, bias_ptrs=
                        bias_ptrs if ATTN_BIAS_TYPE == 'separate' else None,
                        attn_scale=attn_scale if HAS_ATTN_SCALE else None,
                        contextual_seq_len=contextual_seq_len,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE, CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN, ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS, USE_POS_BIAS=
                        USE_POS_BIAS, HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_ATTN_SCALE=HAS_ATTN_SCALE, HAS_MAX_ATTN_LEN=
                        HAS_MAX_ATTN_LEN, HAS_CONTEXTUAL_SEQ_LEN=
                        HAS_CONTEXTUAL_SEQ_LEN, IS_DELTA_Q=IS_DELTA_Q,
                        ALLOW_TF32=ALLOW_TF32, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
                        )
                    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if IS_DELTA_Q:
            start_m_delta = pid * BLOCK_M
            offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = (off_z * DeltaSize + offs_m_delta[:, None]
                ) * stride_om + off_h * stride_oh + offs_v_d[None, :]
            out_ptrs = Out + off_o
            tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
        else:
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = (seq_start + offs_m[:, None]
                ) * stride_om + off_h * stride_oh + offs_v_d[None, :]
            out_ptrs = Out + off_o
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])
