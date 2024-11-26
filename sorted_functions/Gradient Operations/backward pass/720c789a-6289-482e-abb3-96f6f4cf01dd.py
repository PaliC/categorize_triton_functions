import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_recurrence(S, p2, DS, Dp2, NUM_BLOCK, NUM_SPLIT_K, NUM_SPLIT_V,
    D_MODEL_K: 'tl.constexpr', D_MODEL_V: 'tl.constexpr', BLOCK_MODEL:
    'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)
    S = (S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d *
        D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] *
        D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None,
        :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V)
    DS = (DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d *
        D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] *
        D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None,
        :] + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V)
    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL_V + tl.arange(0, BLOCK_MODEL
        ) + offset_s * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_V
    Dp2 = (Dp2 + offset_bh * NUM_BLOCK * D_MODEL_V * NUM_SPLIT_K + offset_d *
        D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + (
        NUM_BLOCK - 2) * D_MODEL_V * NUM_SPLIT_K)
    Dacc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1):
        p_value = tl.load(p2)
        S_i = tl.load(S)
        DS_i = tl.load(DS)
        Dacc += DS_i
        dp_i = Dacc * S_i
        dp_value = tl.sum(dp_i, axis=0)
        tl.store(Dp2, dp_value)
        tl.store(S, Dacc)
        Dacc *= p_value[None, :]
        S -= D_MODEL_K * D_MODEL_V
        DS -= D_MODEL_K * D_MODEL_V
        p2 -= D_MODEL_V
        Dp2 -= D_MODEL_V * NUM_SPLIT_K
