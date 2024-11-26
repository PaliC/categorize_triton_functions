import triton
import triton.language as tl
import torch

@triton.jit
def _attn_bias_bwd(Q, K, V, seq_offsets, TS, TW, PW, num_targets, DOut, DTW,
    DPW, stride_qm, stride_qh, stride_kn, stride_kh, stride_vn, stride_vh,
    stride_ts, stride_dom, stride_doh, alpha, Z, H, MAX_SEQ_LEN, DimQ, DimV,
    num_buckets, max_pos_ind, time_bucket_incr, time_bucket_div, time_delta,
    MAX_ATTN_LEN: 'tl.constexpr', INVALID_MASK_TYPE: 'tl.constexpr', CAUSAL:
    'tl.constexpr', BUCKET_FN: 'tl.constexpr', USE_TIME_BIAS:
    'tl.constexpr', USE_POS_BIAS: 'tl.constexpr', HAS_MAX_POS_IND:
    'tl.constexpr', HAS_MULTIPLE_TARGETS: 'tl.constexpr',
    CONTEXTUAL_SEQ_LEN: 'tl.constexpr', ALLOW_TF32: 'tl.constexpr',
    BLOCK_D_Q: 'tl.constexpr', BLOCK_D_V: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', NUM_N_BLOCKS: 'tl.constexpr', NUM_OUT_GROUPS:
    'tl.constexpr'):
    off_mn = tl.program_id(0)
    off_m = off_mn // NUM_N_BLOCKS
    off_n = off_mn % NUM_N_BLOCKS
    widx = off_m * (off_m + 1) // 2 + off_n
    widx = widx % NUM_OUT_GROUPS
    start_m = off_m * BLOCK_N
    start_n = off_n * BLOCK_N
    offs_m = start_m + tl.arange(0, BLOCK_N)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    dbias_pos = None
    offs_pos_w = None
    if USE_POS_BIAS:
        if not HAS_MULTIPLE_TARGETS:
            dbias_pos = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
            if HAS_MAX_POS_IND:
                offs_pos_w = offs_n[None, :] - offs_m[:, None
                    ] + max_pos_ind - 1
                offs_pos_w = tl.where(offs_pos_w > 0, offs_pos_w, 0)
                offs_pos_w = tl.where(offs_pos_w < 2 * max_pos_ind - 2,
                    offs_pos_w, 2 * max_pos_ind - 2)
            else:
                offs_pos_w = offs_n[None, :] - offs_m[:, None
                    ] + MAX_SEQ_LEN - 1
    if HAS_MULTIPLE_TARGETS:
        invalid_mask = offs_m[:, None] == offs_n[None, :]
    elif MAX_ATTN_LEN > 0:
        if INVALID_MASK_TYPE == 'lower_triangular':
            invalid_mask = offs_m[:, None] >= offs_n[None, :] and offs_m[:,
                None] - offs_n[None, :] <= MAX_ATTN_LEN
        elif INVALID_MASK_TYPE == 'upper_triangular':
            invalid_mask = offs_m[:, None] <= offs_n[None, :] and offs_n[
                None, :] - offs_m[:, None] <= MAX_ATTN_LEN
    elif INVALID_MASK_TYPE == 'lower_triangular':
        invalid_mask = offs_m[:, None] >= offs_n[None, :]
    elif INVALID_MASK_TYPE == 'upper_triangular':
        invalid_mask = offs_m[:, None] <= offs_n[None, :]
    for off_z in range(Z):
        seq_start = tl.load(seq_offsets + off_z)
        seq_end = tl.load(seq_offsets + off_z + 1)
        seq_len = seq_end - seq_start
        if INVALID_MASK_TYPE == 'lower_triangular':
            if HAS_MULTIPLE_TARGETS:
                low = start_n
                if MAX_ATTN_LEN > 0:
                    n_targets = tl.load(num_targets + off_z)
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
        elif INVALID_MASK_TYPE == 'upper_triangular':
            low = 0
            high = start_n + BLOCK_N
        if start_n < seq_len and (start_m >= low and start_m < high):
            q_ptrs = Q + seq_start * stride_qm + offs_m[:, None
                ] * stride_qm + offs_qk_d[None, :]
            k_ptrs = K + seq_start * stride_kn + offs_n[:, None
                ] * stride_kn + offs_qk_d[None, :]
            v_ptrs = V + seq_start * stride_vn + offs_n[:, None
                ] * stride_vn + offs_v_d[None, :]
            do_ptrs = DOut + seq_start * stride_dom + offs_m[:, None
                ] * stride_dom + offs_v_d[None, :]
            mask_m = offs_m < seq_len
            mask_n = offs_n < seq_len
            if HAS_MULTIPLE_TARGETS:
                if (INVALID_MASK_TYPE != 'lower_triangular' or MAX_ATTN_LEN ==
                    0):
                    n_targets = tl.load(num_targets + off_z)
                if INVALID_MASK_TYPE == 'lower_triangular':
                    pos_offs_m = tl.where(offs_m < seq_len - n_targets,
                        offs_m, seq_len - n_targets)
                    pos_offs_n = tl.where(offs_n < seq_len - n_targets,
                        offs_n, seq_len - n_targets)
                elif INVALID_MASK_TYPE == 'upper_triangular':
                    pos_offs_m = tl.where(offs_m > n_targets - 1, offs_m, 
                        n_targets - 1)
                    pos_offs_n = tl.where(offs_n > n_targets - 1, offs_n, 
                        n_targets - 1)
            else:
                pos_offs_n = offs_n
                pos_offs_m = offs_m
            mt_offs_pos_w = None
            if USE_POS_BIAS:
                if HAS_MULTIPLE_TARGETS:
                    if HAS_MAX_POS_IND:
                        mt_offs_pos_w = pos_offs_n[None, :] - pos_offs_m[:,
                            None] + max_pos_ind - 1
                        mt_offs_pos_w = tl.where(mt_offs_pos_w > 0,
                            mt_offs_pos_w, 0)
                        mt_offs_pos_w = tl.where(mt_offs_pos_w < 2 *
                            max_pos_ind - 2, mt_offs_pos_w, 2 * max_pos_ind - 2
                            )
                    else:
                        mt_offs_pos_w = pos_offs_n[None, :] - pos_offs_m[:,
                            None] + MAX_SEQ_LEN - 1
                else:
                    mt_offs_pos_w = offs_pos_w
            if HAS_MULTIPLE_TARGETS:
                if MAX_ATTN_LEN > 0:
                    if INVALID_MASK_TYPE == 'lower_triangular':
                        mt_invalid_mask = invalid_mask or pos_offs_m[:, None
                            ] > pos_offs_n[None, :] and pos_offs_n[None, :
                            ] - pos_offs_m[:, None] >= -MAX_ATTN_LEN
                    elif INVALID_MASK_TYPE == 'upper_triangular':
                        mt_invalid_mask = invalid_mask or pos_offs_m[:, None
                            ] < pos_offs_n[None, :] and pos_offs_n[None, :
                            ] - pos_offs_m[:, None] <= MAX_ATTN_LEN
                elif INVALID_MASK_TYPE == 'lower_triangular':
                    mt_invalid_mask = invalid_mask or pos_offs_m[:, None
                        ] > pos_offs_n[None, :]
                elif INVALID_MASK_TYPE == 'upper_triangular':
                    mt_invalid_mask = invalid_mask or pos_offs_m[:, None
                        ] < pos_offs_n[None, :]
            else:
                mt_invalid_mask = invalid_mask
            if CONTEXTUAL_SEQ_LEN > 0:
                if INVALID_MASK_TYPE == 'lower_triangular':
                    row_filter = offs_m < CONTEXTUAL_SEQ_LEN
                    if HAS_MULTIPLE_TARGETS:
                        col_filter = offs_n < seq_len - n_targets
                    else:
                        col_filter = offs_n < seq_len
                    invalid_mask = invalid_mask or row_filter[:, None
                        ] and col_filter[None, :]
            ts = None
            if USE_TIME_BIAS:
                ts_ptrs = TS + off_z * stride_ts
                ts_0_ptrs = ts_ptrs + offs_m
                ts_1_ptrs = ts_ptrs + offs_n
                if CAUSAL:
                    ts_0 = tl.load(ts_0_ptrs + 1, mask=mask_m)
                    ts_1 = tl.load(ts_1_ptrs, mask=mask_n)
                else:
                    ts_0 = tl.load(ts_0_ptrs, mask=mask_m)
                    ts_1 = tl.load(ts_1_ptrs + 1, mask=mask_n)
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
            attn_bias = tl.zeros([BLOCK_N, BLOCK_N], dtype=tl.float32)
            if USE_TIME_BIAS:
                ts_w = tl.load(TW + ts, mask=mask_m[:, None] & mask_n[None,
                    :] & mt_invalid_mask)
                attn_bias = attn_bias + ts_w
            if USE_POS_BIAS:
                pos_w = tl.load(PW + mt_offs_pos_w, mask=mask_m[:, None] &
                    mask_n[None, :] & mt_invalid_mask)
                attn_bias = attn_bias + pos_w
            dbias = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
            for off_h in range(H):
                q = tl.load(q_ptrs + off_h * stride_qh, mask=mask_m[:, None
                    ], other=0.0)
                k = tl.load(k_ptrs + off_h * stride_kh, mask=mask_n[:, None
                    ], other=0.0)
                qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * alpha
                qk = qk + attn_bias
                sig = fast_dividef(1.0, 1.0 + tl.exp(-qk))
                do = tl.load(do_ptrs + off_h * stride_doh, mask=mask_m[:,
                    None], other=0.0)
                v = tl.load(v_ptrs + off_h * stride_vh, mask=mask_n[:, None
                    ], other=0.0)
                dqk = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)
                dqk = dqk * sig * (1 + qk * (1 - sig)) * (1.0 / MAX_SEQ_LEN)
                dbias = dbias + dqk
            if USE_TIME_BIAS:
                dtw_ptrs = DTW + widx * (num_buckets + 1)
                tl.atomic_add(dtw_ptrs + ts, dbias, mask=mask_m[:, None] &
                    mask_n[None, :] & mt_invalid_mask, sem='relaxed')
            if USE_POS_BIAS:
                if HAS_MULTIPLE_TARGETS:
                    if HAS_MAX_POS_IND:
                        dpw_ptrs = DPW + widx * (2 * max_pos_ind - 1)
                    else:
                        dpw_ptrs = DPW + widx * (2 * MAX_SEQ_LEN - 1)
                    tl.atomic_add(dpw_ptrs + mt_offs_pos_w, dbias, mask=
                        mask_m[:, None] & mask_n[None, :] & mt_invalid_mask,
                        sem='relaxed')
                else:
                    dbias_pos += dbias
    if USE_POS_BIAS and not HAS_MULTIPLE_TARGETS:
        if HAS_MAX_POS_IND:
            dpw_ptrs = DPW + widx * (2 * max_pos_ind - 1)
        else:
            dpw_ptrs = DPW + widx * (2 * MAX_SEQ_LEN - 1)
        tl.atomic_add(dpw_ptrs + offs_pos_w, dbias_pos, mask=invalid_mask,
            sem='relaxed')
