import triton
import triton.language as tl
import torch

@triton.autotune(configs=TRITON_CONFIG_LIST_BWD_FUSED, key=['max_seqlen_q',
    'max_seqlen_k', 'head_dim'])
@triton.jit
def tuned_attn_bwd(Q, K, V, B, sm_scale, Out, DO, DK, DV, DQ, DB, L, D,
    stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
    stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn, stride_oz, stride_oh,
    stride_om, stride_ok, stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn, stride_dqz, stride_dqh,
    stride_dqm, stride_dqk, stride_dbz, stride_dbh, stride_dbm, stride_dbn,
    num_head_q, num_head_k, cu_seqlens_q, cu_seqlens_k, num_seqlens,
    max_seqlen_q, max_seqlen_k, head_dim, dropout_p, philox_seed_ptr,
    philox_offset1, philox_offset2, BLOCK_DMODEL: 'tl.constexpr', CAUSAL:
    'tl.constexpr', ENABLE_DROPOUT: 'tl.constexpr', PADDED_HEAD:
    'tl.constexpr', BIAS_TYPE: 'tl.constexpr', BLOCK_M1: 'tl.constexpr',
    BLOCK_N1: 'tl.constexpr', BLOCK_M2: 'tl.constexpr', BLOCK_N2:
    'tl.constexpr', BLK_SLICE_FACTOR: 'tl.constexpr'):
    bare_attn_bwd(Q, K, V, B, sm_scale, Out, DO, DK, DV, DQ, DB, L, D,
        stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
        stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn,
        stride_bz, stride_bh, stride_bm, stride_bn, stride_oz, stride_oh,
        stride_om, stride_ok, stride_dkz, stride_dkh, stride_dkn,
        stride_dkk, stride_dvz, stride_dvh, stride_dvk, stride_dvn,
        stride_dqz, stride_dqh, stride_dqm, stride_dqk, stride_dbz,
        stride_dbh, stride_dbm, stride_dbn, num_head_q, num_head_k,
        cu_seqlens_q, cu_seqlens_k, num_seqlens, max_seqlen_q, max_seqlen_k,
        head_dim, dropout_p, philox_seed_ptr, philox_offset_base,
        BLOCK_DMODEL, CAUSAL, ENABLE_DROPOUT, PADDED_HEAD, BIAS_TYPE,
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, BLK_SLICE_FACTOR)
