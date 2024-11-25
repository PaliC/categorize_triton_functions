import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_preprocess_cumsum_gk(Q, K, GK, GK_cumsum, DQ_exp, DK_reduce,
    DGK_last_exp, DGK_cumsum, DQ, DK, DGK, NUM_CHUNK, L, D_MODEL_K:
    'tl.constexpr', D_BLOCK_K: 'tl.constexpr', CHUNK_SIZE: 'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    offset_nk = tl.program_id(2)
    mask = D_BLOCK_K * offset_nk + tl.arange(0, D_BLOCK_K) < D_MODEL_K
    Q_ptr = (Q + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    K_ptr = (K + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    GK_ptr = (GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    GK_cumsum_ptr = (GK_cumsum + offset_bh * L * D_MODEL_K + offset_c *
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K *
        offset_nk)
    DQ_ptr = (DQ + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    DK_ptr = (DK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    DQ_exp_ptr = (DQ_exp + offset_bh * L * D_MODEL_K + offset_c *
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K *
        offset_nk)
    DK_reduce_ptr = (DK_reduce + offset_bh * L * D_MODEL_K + offset_c *
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K *
        offset_nk)
    DGK_cumsum_ptr = (DGK_cumsum + offset_bh * L * D_MODEL_K + offset_c *
        CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K *
        offset_nk)
    DGK_ptr = (DGK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    D_GK_last_exp_ptr = (DGK_last_exp + offset_bh * NUM_CHUNK * D_MODEL_K +
        offset_c * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk)
    cumsum_gradient = tl.zeros([D_BLOCK_K], dtype=tl.float32)
    grad_gk_last = tl.zeros([D_BLOCK_K], dtype=tl.float32)
    gk_last = tl.load(GK_cumsum_ptr + (CHUNK_SIZE - 1) * D_MODEL_K, mask=
        mask, other=0)
    cumsum_gradient += tl.load(D_GK_last_exp_ptr, mask=mask, other=0) * tl.exp(
        gk_last)
    GK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    GK_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    Q_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    K_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DQ_exp_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DK_reduce_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DGK_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DQ_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DGK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    for idx in range(CHUNK_SIZE - 1, -1, -1):
        gk_cs = tl.load(GK_cumsum_ptr, mask=mask, other=0)
        k = tl.load(K_ptr, mask=mask, other=0)
        grad_k = tl.exp(gk_last - gk_cs) * tl.load(DK_reduce_ptr, mask=mask,
            other=0)
        tl.store(DK_ptr, grad_k, mask=mask)
        grad_k *= k
        cumsum_gradient -= grad_k
        grad_gk_last += grad_k
        q = tl.load(Q_ptr, mask=mask, other=0)
        grad_q = tl.exp(gk_cs) * tl.load(DQ_exp_ptr, mask=mask, other=0)
        tl.store(DQ_ptr, grad_q, mask=mask)
        cumsum_gradient += grad_q * q
        cumsum_gradient += tl.load(DGK_cumsum_ptr, mask=mask, other=0)
        tl.store(DGK_ptr, cumsum_gradient, mask=mask)
        Q_ptr -= D_MODEL_K
        DQ_exp_ptr -= D_MODEL_K
        K_ptr -= D_MODEL_K
        DK_reduce_ptr -= D_MODEL_K
        GK_cumsum_ptr -= D_MODEL_K
        DGK_cumsum_ptr -= D_MODEL_K
        DQ_ptr -= D_MODEL_K
        DK_ptr -= D_MODEL_K
        DGK_ptr -= D_MODEL_K
    DGK_ptr = (DGK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + (CHUNK_SIZE - 1) * D_MODEL_K +
        D_BLOCK_K * offset_nk)
    GK_ptr = (GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE *
        D_MODEL_K + tl.arange(0, D_BLOCK_K) + (CHUNK_SIZE - 1) * D_MODEL_K +
        D_BLOCK_K * offset_nk)
    grad_gk_last = grad_gk_last + 0.0
    for idx in range(CHUNK_SIZE - 1, -1, -1):
        dgk = tl.load(DGK_ptr, mask=mask, other=0)
        dgk += grad_gk_last
        tl.store(DGK_ptr, dgk, mask=mask)
        DGK_ptr -= D_MODEL_K
        GK_ptr -= D_MODEL_K
