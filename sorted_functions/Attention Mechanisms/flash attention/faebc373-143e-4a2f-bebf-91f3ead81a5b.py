import triton
import triton.language as tl
import torch

@triton.autotune(list(filter(keep, configsOrig)), key=['N_CTX'])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out, desc_q, desc_k, desc_v, desc_o,
    stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
    stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX, BLOCK_M:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', HEAD_DIM: 'tl.constexpr',
    STAGE: 'tl.constexpr', ENABLE_TMA: 'tl.constexpr', LOOP_SCHEDULE:
    'tl.constexpr', ENABLE_WS: 'tl.constexpr'):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    _attn_fwd_compute(Q, K, V, sm_scale, M, Out, desc_q, desc_k, desc_v,
        desc_o, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
        stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk,
        stride_vn, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX,
        BLOCK_M, BLOCK_N, HEAD_DIM, STAGE, ENABLE_TMA, LOOP_SCHEDULE)
