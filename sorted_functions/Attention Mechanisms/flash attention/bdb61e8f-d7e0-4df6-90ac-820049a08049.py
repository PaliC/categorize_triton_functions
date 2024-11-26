import triton
import triton.language as tl
import torch

@triton.autotune(configs=TRITON_CONFIG_LIST_FWD, key=['max_seqlen_q',
    'max_seqlen_k', 'CAUSAL'])
@triton.jit
def tuned_attn_fwd(Q, K, V, B, sm_scale, M, Out, stride_qz, stride_qh,
    stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn, stride_bz, stride_bh,
    stride_bm, stride_bn, stride_oz, stride_oh, stride_om, stride_on,
    num_head_q, num_head_k, cu_seqlens_q, cu_seqlens_k, num_seqlens,
    max_seqlen_q, max_seqlen_k, head_dim, dropout_p, philox_seed_ptr,
    philox_offset1, philox_offset2, philox_seed_output,
    philox_offset_output, encoded_softmax, CAUSAL: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    pre_load_v: 'tl.constexpr', ENABLE_DROPOUT: 'tl.constexpr',
    RETURN_ENCODED_SOFTMAX: 'tl.constexpr', PADDED_HEAD: 'tl.constexpr',
    BIAS_TYPE: 'tl.constexpr'):
    bare_attn_fwd(Q, K, V, B, sm_scale, M, Out, stride_qz, stride_qh,
        stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn, stride_bz, stride_bh,
        stride_bm, stride_bn, stride_oz, stride_oh, stride_om, stride_on,
        num_head_q, num_head_k, cu_seqlens_q, cu_seqlens_k, num_seqlens,
        max_seqlen_q, max_seqlen_k, head_dim, dropout_p, philox_seed_ptr,
        philox_offset1, philox_offset2, philox_seed_output,
        philox_offset_output, encoded_softmax, CAUSAL, BLOCK_M,
        BLOCK_DMODEL, BLOCK_N, pre_load_v, ENABLE_DROPOUT,
        RETURN_ENCODED_SOFTMAX, PADDED_HEAD, BIAS_TYPE=BIAS_TYPE)
