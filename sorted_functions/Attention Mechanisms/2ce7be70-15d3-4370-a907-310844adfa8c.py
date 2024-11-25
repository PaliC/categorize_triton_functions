import triton
import triton.language as tl
import torch

@triton.autotune(configs=_get_fw_configs(), key=['Z', 'H',
    'AUTOTUNE_MAX_SEQ_LEN', 'DimQ', 'DimV', 'BUCKET_FN', 'ATTN_BIAS_TYPE',
    'DeltaSize', 'IS_DELTA_Q'])
@triton.jit
def _ragged_hstu_attn_fwd_persistent(Q, K, V, sort_by_length_indices,
    seq_offsets, TS, TW, PW, Bias, seq2_offsets, delta_x_offsets,
    num_targets, Scale, Out, stride_qm, stride_qh, stride_kn, stride_kh,
    stride_vn, stride_vh, stride_sz, stride_sm, stride_ts, stride_om,
    stride_oh, alpha, Z, H, MAX_SEQ_LEN, AUTOTUNE_MAX_SEQ_LEN, DimQ, DimV,
    DeltaSize, num_buckets, max_pos_ind, time_bucket_incr, time_bucket_div,
    time_delta, contextual_seq_len, INVALID_MASK_TYPE: 'tl.constexpr',
    CAUSAL: 'tl.constexpr', BUCKET_FN: 'tl.constexpr', ATTN_BIAS_TYPE:
    'tl.constexpr', USE_TIME_BIAS: 'tl.constexpr', USE_POS_BIAS:
    'tl.constexpr', HAS_MAX_POS_IND: 'tl.constexpr', HAS_MULTIPLE_TARGETS:
    'tl.constexpr', HAS_ATTN_SCALE: 'tl.constexpr', IS_DELTA_Q:
    'tl.constexpr', ALLOW_TF32: 'tl.constexpr', BLOCK_D_Q: 'tl.constexpr',
    BLOCK_D_V: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', max_attn_len: 'tl.constexpr', HAS_MAX_ATTN_LEN:
    'tl.constexpr', HAS_CONTEXTUAL_SEQ_LEN: 'tl.constexpr',
    HAS_SORT_BY_LENGTH_INDICES: 'tl.constexpr'):
    n_tile_num = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H
    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1
    tile_idx = prog_id
    for _ in range(0, tiles_per_sm):
        pid = (total_tiles - tile_idx - 1) // (Z * H)
        off_hz = (total_tiles - tile_idx - 1) % (Z * H)
        off_z = off_hz // H
        off_h = off_hz % H
        _ragged_hstu_attn_fwd_compute(Q=Q, K=K, V=V, seq_offsets=
            seq_offsets, TS=TS, TW=TW, PW=PW, Bias=Bias, seq2_offsets=
            seq2_offsets, delta_x_offsets=delta_x_offsets, num_targets=
            num_targets, Scale=Scale, Out=Out, stride_qm=stride_qm,
            stride_qh=stride_qh, stride_kn=stride_kn, stride_kh=stride_kh,
            stride_vn=stride_vn, stride_vh=stride_vh, stride_sz=stride_sz,
            stride_sm=stride_sm, stride_ts=stride_ts, stride_om=stride_om,
            stride_oh=stride_oh, alpha=alpha, Z=Z, H=H, MAX_SEQ_LEN=
            MAX_SEQ_LEN, DimQ=DimQ, DimV=DimV, DeltaSize=DeltaSize,
            num_buckets=num_buckets, max_pos_ind=max_pos_ind,
            time_bucket_incr=time_bucket_incr, time_bucket_div=
            time_bucket_div, time_delta=time_delta, contextual_seq_len=
            contextual_seq_len, off_z=off_z, off_h=off_h, pid=pid,
            INVALID_MASK_TYPE=INVALID_MASK_TYPE, CAUSAL=CAUSAL, BUCKET_FN=
            BUCKET_FN, ATTN_BIAS_TYPE=ATTN_BIAS_TYPE, USE_TIME_BIAS=
            USE_TIME_BIAS, USE_POS_BIAS=USE_POS_BIAS, HAS_MAX_POS_IND=
            HAS_MAX_POS_IND, HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_ATTN_SCALE=HAS_ATTN_SCALE, IS_DELTA_Q=IS_DELTA_Q,
            ALLOW_TF32=ALLOW_TF32, BLOCK_D_Q=BLOCK_D_Q, BLOCK_D_V=BLOCK_D_V,
            max_attn_len=max_attn_len, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN, BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N)
        tile_idx += num_progs
