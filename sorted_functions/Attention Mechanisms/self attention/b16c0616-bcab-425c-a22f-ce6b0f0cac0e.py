import triton
import triton.language as tl
import torch

@triton.jit
def _ragged_hstu_attn_bwd_one_block(start_m, offs_n, offs_m, q_ptrs_trans,
    dq_ptrs_trans, mask_n, ts_0_ptrs, ts_1, bias_ptrs_trans,
    dbias_ptrs_trans, do_ptrs, dk, dv, k, v, pos_offs_n, seq_len, n_targets,
    TW, PW, DTW, DPW, LOCK, stride_qm, stride_dom, stride_dqm, alpha,
    MAX_SEQ_LEN, num_buckets, max_pos_ind, time_bucket_incr,
    time_bucket_div, time_delta, MAX_ATTN_LEN: 'tl.constexpr',
    INVALID_MASK_TYPE: 'tl.constexpr', CAUSAL: 'tl.constexpr', BUCKET_FN:
    'tl.constexpr', ATTN_BIAS_TYPE: 'tl.constexpr', USE_TIME_BIAS:
    'tl.constexpr', USE_POS_BIAS: 'tl.constexpr', FUSED_BIAS_BWD:
    'tl.constexpr', HAS_MAX_POS_IND: 'tl.constexpr', HAS_MULTIPLE_TARGETS:
    'tl.constexpr', CONTEXTUAL_SEQ_LEN: 'tl.constexpr', ALLOW_TF32:
    'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    ATOMIC_ADD: 'tl.constexpr'):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
    if HAS_MULTIPLE_TARGETS:
        if INVALID_MASK_TYPE == 'lower_triangular':
            pos_offs_m = tl.where(pos_offs_m < seq_len - n_targets,
                pos_offs_m, seq_len - n_targets)
        elif INVALID_MASK_TYPE == 'upper_triangular':
            pos_offs_m = tl.where(pos_offs_m > n_targets - 1, pos_offs_m, 
                n_targets - 1)
    q_trans = tl.load(q_ptrs_trans + start_m * stride_qm, mask=mask_m[None,
        :], other=0.0)
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32) * alpha
    if ATTN_BIAS_TYPE == 'fused':
        attn_bias_trans = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        if USE_TIME_BIAS:
            if CAUSAL:
                ts_0 = tl.load(ts_0_ptrs + start_m + 1, mask=mask_m)
            else:
                ts_0 = tl.load(ts_0_ptrs + start_m, mask=mask_m)
            ts_trans = ts_0[None, :] - ts_1[:, None]
            ts_trans = ts_trans + time_delta
            ts_trans = tl.where(ts_trans > 1e-06, ts_trans, 1e-06)
            ts_trans = ts_trans * (1.0 / time_bucket_incr)
            if BUCKET_FN == 'log':
                ts_trans = tl.log(ts_trans)
            elif BUCKET_FN == 'sqrt':
                ts_trans = tl.sqrt(ts_trans)
            ts_trans = ts_trans * (1.0 / time_bucket_div)
            ts_trans = ts_trans
            ts_trans = tl.where(ts_trans > 0, ts_trans, 0)
            ts_trans = tl.where(ts_trans < num_buckets, ts_trans, num_buckets)
            ts_w_trans = tl.load(TW + ts_trans, mask=mask_m[None, :] and
                mask_n[:, None])
            attn_bias_trans = attn_bias_trans + ts_w_trans
        if USE_POS_BIAS:
            offs_pos_w_trans = None
            if HAS_MAX_POS_IND:
                offs_pos_w_trans = pos_offs_n[:, None] - pos_offs_m[None, :
                    ] + max_pos_ind - 1
                offs_pos_w_trans = tl.where(offs_pos_w_trans > 0,
                    offs_pos_w_trans, 0)
                offs_pos_w_trans = tl.where(offs_pos_w_trans < 2 *
                    max_pos_ind - 2, offs_pos_w_trans, 2 * max_pos_ind - 2)
            else:
                offs_pos_w_trans = pos_offs_n[:, None] - pos_offs_m[None, :
                    ] + MAX_SEQ_LEN - 1
            pos_w_trans = tl.load(PW + offs_pos_w_trans, mask=mask_m[None,
                :] and mask_n[:, None])
            attn_bias_trans = attn_bias_trans + pos_w_trans
        qk_trans = qk_trans + attn_bias_trans
    elif ATTN_BIAS_TYPE == 'separate':
        attn_bias_trans = tl.load(bias_ptrs_trans + start_m * seq_len, mask
            =mask_m[None, :] & mask_n[:, None], other=0.0)
        qk_trans = qk_trans + attn_bias_trans
    sig_trans = fast_dividef(1.0, 1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    if MAX_ATTN_LEN > 0:
        if INVALID_MASK_TYPE == 'lower_triangular':
            invalid_mask_trans = invalid_mask_trans or pos_offs_m[None, :
                ] > pos_offs_n[:, None] and pos_offs_n[:, None] - pos_offs_m[
                None, :] >= -MAX_ATTN_LEN
        elif INVALID_MASK_TYPE == 'upper_triangular':
            invalid_mask_trans = invalid_mask_trans or pos_offs_m[None, :
                ] < pos_offs_n[:, None] and pos_offs_n[:, None] - pos_offs_m[
                None, :] <= MAX_ATTN_LEN
    elif INVALID_MASK_TYPE == 'lower_triangular':
        invalid_mask_trans = invalid_mask_trans or pos_offs_m[None, :
            ] > pos_offs_n[:, None]
    elif INVALID_MASK_TYPE == 'upper_triangular':
        invalid_mask_trans = invalid_mask_trans or pos_offs_m[None, :
            ] < pos_offs_n[:, None]
    if CONTEXTUAL_SEQ_LEN > 0 and INVALID_MASK_TYPE == 'lower_triangular':
        row_filter = pos_offs_m < CONTEXTUAL_SEQ_LEN
        if HAS_MULTIPLE_TARGETS:
            col_filter = pos_offs_n < seq_len - n_targets
        else:
            col_filter = pos_offs_n < seq_len
        invalid_mask_trans = invalid_mask_trans or row_filter[None, :
            ] and col_filter[:, None]
    silu_trans = tl.where(invalid_mask_trans, silu_trans, 0)
    silu_trans = silu_trans
    do = tl.load(do_ptrs + start_m * stride_dom, mask=mask_m[:, None],
        other=0.0)
    dv += tl.dot(silu_trans, do, allow_tf32=ALLOW_TF32)
    dqk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
    dqk_trans = dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * (
        1.0 / MAX_SEQ_LEN)
    dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
    dqk_trans = dqk_trans
    if ATTN_BIAS_TYPE == 'fused' and FUSED_BIAS_BWD:
        if USE_TIME_BIAS:
            tl.atomic_add(DTW + ts_trans, dqk_trans, mask=mask_m[None, :] &
                mask_n[:, None] & invalid_mask_trans, sem='relaxed')
        if USE_POS_BIAS:
            tl.atomic_add(DPW + offs_pos_w_trans, dqk_trans, mask=mask_m[
                None, :] & mask_n[:, None] & invalid_mask_trans, sem='relaxed')
    elif ATTN_BIAS_TYPE == 'separate':
        tl.store(dbias_ptrs_trans + start_m * seq_len, dqk_trans, mask=
            mask_m[None, :] & mask_n[:, None])
    dk += tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32)
    if ATOMIC_ADD:
        lock_id = start_m // BLOCK_M
        stride_lock = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
        lock = LOCK + tl.program_id(0) * stride_lock + lock_id
        tl.debug_barrier()
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass
    dq_trans = tl.load(dq_ptrs_trans + start_m * stride_dqm, mask=mask_m[
        None, :], other=0.0, eviction_policy='evict_last')
    dq_trans += tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha
    dq_trans = dq_trans
    tl.store(dq_ptrs_trans + start_m * stride_dqm, dq_trans, mask=mask_m[
        None, :], eviction_policy='evict_last')
    if ATOMIC_ADD:
        tl.atomic_xchg(lock, 0)
    return dk, dv
