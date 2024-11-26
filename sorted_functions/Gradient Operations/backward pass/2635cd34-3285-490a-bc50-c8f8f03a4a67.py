import triton
import triton.language as tl
import torch

@triton.autotune(configs=TRITON_CONFIG_LIST_BWD, key=['BLOCK_DMODEL',
    'max_seqlen_q', 'max_seqlen_k'])
@triton.jit
def tuned_bwd_kernel_dq(Q, K, V, B, sm_scale, Out, DO, DQ, DB, L, D,
    stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
    stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn, stride_oz, stride_oh,
    stride_om, stride_ok, stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dbz, stride_dbh, stride_dbm, stride_dbn, cu_seqlens_q,
    cu_seqlens_k, num_seqlens, max_seqlen_q, max_seqlen_k, head_dim,
    dropout_p, philox_seed, philox_offset_base, BLOCK_M: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', CAUSAL:
    'tl.constexpr', ENABLE_DROPOUT: 'tl.constexpr', PADDED_HEAD:
    'tl.constexpr', BIAS_TYPE: 'tl.constexpr'):
    bare_bwd_kernel_dq(Q, K, V, B, sm_scale, Out, DO, DQ, DB, L, D,
        stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
        stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn,
        stride_bz, stride_bh, stride_bm, stride_bn, stride_oz, stride_oh,
        stride_om, stride_ok, stride_dqz, stride_dqh, stride_dqm,
        stride_dqk, stride_dbz, stride_dbh, stride_dbm, stride_dbn,
        cu_seqlens_q, cu_seqlens_k, num_seqlens, max_seqlen_q, max_seqlen_k,
        head_dim, dropout_p, philox_seed, philox_offset_base, BLOCK_M,
        BLOCK_DMODEL, BLOCK_N, CAUSAL, ENABLE_DROPOUT, PADDED_HEAD=
        PADDED_HEAD, BIAS_TYPE=BIAS_TYPE)
