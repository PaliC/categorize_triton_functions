import triton
import triton.language as tl
import torch

@triton.jit
def _ragged_hstu_attn_fwd_one_block(start_n, seq_len, offs_m, offs_n,
    mask_m, mask_n, q, K_block_ptr, V_block_ptr, n_targets, ts_1_ptrs, ts_0,
    TW, PW, alpha, MAX_SEQ_LEN, num_buckets, max_pos_ind, max_attn_len,
    time_bucket_incr, time_bucket_div, time_delta, bias_ptrs, attn_scale,
    contextual_seq_len, INVALID_MASK_TYPE: 'tl.constexpr', CAUSAL:
    'tl.constexpr', BUCKET_FN: 'tl.constexpr', ATTN_BIAS_TYPE:
    'tl.constexpr', USE_TIME_BIAS: 'tl.constexpr', USE_POS_BIAS:
    'tl.constexpr', HAS_MAX_POS_IND: 'tl.constexpr', HAS_MULTIPLE_TARGETS:
    'tl.constexpr', HAS_ATTN_SCALE: 'tl.constexpr', HAS_MAX_ATTN_LEN:
    'tl.constexpr', HAS_CONTEXTUAL_SEQ_LEN: 'tl.constexpr', IS_DELTA_Q:
    'tl.constexpr', ALLOW_TF32: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr'):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    k = tl.load(K_block_ptr, boundary_check=(1,), padding_option='zero')
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    if HAS_MULTIPLE_TARGETS:
        if INVALID_MASK_TYPE == 'lower_triangular':
            offs_m = tl.where(offs_m < seq_len - n_targets, offs_m, seq_len -
                n_targets)
            offs_n = tl.where(offs_n < seq_len - n_targets, offs_n, seq_len -
                n_targets)
        elif INVALID_MASK_TYPE == 'upper_triangular':
            offs_m = tl.where(offs_m > n_targets - 1, offs_m, n_targets - 1)
            offs_n = tl.where(offs_n > n_targets - 1, offs_n, n_targets - 1)
    offs_n_minus_m = offs_n[None, :] - offs_m[:, None]
    if HAS_MAX_ATTN_LEN:
        if INVALID_MASK_TYPE == 'lower_triangular':
            invalid_mask = (invalid_mask or offs_n_minus_m < 0 and 
                offs_n_minus_m >= -max_attn_len)
        elif INVALID_MASK_TYPE == 'upper_triangular':
            invalid_mask = (invalid_mask or offs_n_minus_m > 0 and 
                offs_n_minus_m <= max_attn_len)
    elif INVALID_MASK_TYPE == 'lower_triangular':
        invalid_mask = invalid_mask or offs_n_minus_m < 0
    elif INVALID_MASK_TYPE == 'upper_triangular':
        invalid_mask = invalid_mask or offs_n_minus_m > 0
    if HAS_CONTEXTUAL_SEQ_LEN:
        if INVALID_MASK_TYPE == 'lower_triangular':
            row_filter = offs_m < contextual_seq_len
            if HAS_MULTIPLE_TARGETS:
                col_filter = offs_n < seq_len - n_targets
            else:
                col_filter = offs_n < seq_len
            invalid_mask = invalid_mask or row_filter[:, None] and col_filter[
                None, :]
    if ATTN_BIAS_TYPE == 'fused':
        attn_bias = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if USE_TIME_BIAS:
            if CAUSAL:
                ts_1 = tl.load(ts_1_ptrs + start_n, mask=mask_n)
            else:
                ts_1 = tl.load(ts_1_ptrs + start_n + 1, mask=mask_n)
            ts = ts_0[:, None] - ts_1[None, :]
            ts = ts + time_delta
            ts = tl.where(ts > 1e-06, ts, 1e-06)
            ts = ts * (1.0 / time_bucket_incr)
            if BUCKET_FN == 'log':
                ts = tl.log(ts)
            elif BUCKET_FN == 'sqrt':
                ts = tl.sqrt(ts)
            ts = ts * (1.0 / time_bucket_div)
            ts = ts
            ts = tl.where(ts > 0, ts, 0)
            ts = tl.where(ts < num_buckets, ts, num_buckets)
            ts_w = tl.load(TW + ts, mask=mask_m[:, None] and mask_n[None, :])
            attn_bias = attn_bias + ts_w
        if USE_POS_BIAS:
            if HAS_MAX_POS_IND:
                offs_pos_w = offs_n_minus_m + max_pos_ind - 1
                offs_pos_w = tl.where(offs_pos_w > 0, offs_pos_w, 0)
                offs_pos_w = tl.where(offs_pos_w < 2 * max_pos_ind - 2,
                    offs_pos_w, 2 * max_pos_ind - 2)
            else:
                offs_pos_w = offs_n_minus_m + MAX_SEQ_LEN - 1
            pos_w = tl.load(PW + offs_pos_w, mask=mask_m[:, None] and
                mask_n[None, :])
            attn_bias = attn_bias + pos_w
        qk = qk + attn_bias
    elif ATTN_BIAS_TYPE == 'separate':
        attn_bias = tl.load(bias_ptrs + start_n, mask=mask_m[:, None] &
            mask_n[None, :], other=0.0)
        qk = qk + attn_bias
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)
    silu = tl.where(invalid_mask, silu, 0)
    if HAS_ATTN_SCALE:
        silu = silu * attn_scale[:, None]
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
    silu = silu
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)
