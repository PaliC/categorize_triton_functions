import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_preprocess_cumsum_gk(Q, K, GK, GK_cumsum, Q_exp, K_reduce,
    GK_last_exp, NUM_CHUNK, L, D_MODEL_K: 'tl.constexpr', D_BLOCK_K:
    'tl.constexpr', CHUNK_SIZE: 'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    offset_nk = tl.program_id(2)
    Q_ptr = (Q + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    Q_exp_ptr = (Q_exp + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    GK_ptr = (GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    GK_cumsum_ptr = (GK_cumsum + offset_bh * L * D_MODEL_K + offset_c *
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K *
        offset_nk)
    GK_last_exp_ptr = (GK_last_exp + offset_bh * NUM_CHUNK * D_MODEL_K + 
        offset_c * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    cumsum = tl.zeros([D_BLOCK_K], dtype=tl.float32)
    mask = D_BLOCK_K * offset_nk + tl.arange(0, D_BLOCK_K) < D_MODEL_K
    for _ in range(CHUNK_SIZE):
        gk = tl.load(GK_ptr, mask=mask, other=0)
        cumsum += gk
        tl.store(GK_cumsum_ptr, cumsum, mask=mask)
        cumsum_exp = tl.exp(cumsum)
        q = tl.load(Q_ptr, mask=mask, other=0)
        q_exp = q * cumsum_exp
        tl.store(Q_exp_ptr, q_exp, mask=mask)
        Q_ptr += D_MODEL_K
        Q_exp_ptr += D_MODEL_K
        GK_ptr += D_MODEL_K
        GK_cumsum_ptr += D_MODEL_K
    tl.store(GK_last_exp_ptr, tl.exp(cumsum), mask=mask)
    tl.debug_barrier()
    GK_cumsum_ptr = (GK_cumsum + offset_bh * L * D_MODEL_K + offset_c *
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K *
        offset_nk)
    K_ptr = (K + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    K_reduce_ptr = (K_reduce + offset_bh * L * D_MODEL_K + offset_c *
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K *
        offset_nk)
    for _ in range(CHUNK_SIZE):
        gk_cumsum = tl.load(GK_cumsum_ptr, mask=mask, other=0)
        k = tl.load(K_ptr, mask=mask, other=0)
        k_reduce = k * tl.exp(cumsum - gk_cumsum)
        tl.store(K_reduce_ptr, k_reduce, mask=mask)
        K_ptr += D_MODEL_K
        GK_cumsum_ptr += D_MODEL_K
        K_reduce_ptr += D_MODEL_K
