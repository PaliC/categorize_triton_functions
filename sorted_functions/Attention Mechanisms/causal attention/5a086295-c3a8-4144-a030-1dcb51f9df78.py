import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_recurrence(S, p2, O, NUM_BLOCK, D_MODEL_K: 'tl.constexpr',
    D_MODEL_V: 'tl.constexpr', BLOCK_MODEL: 'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)
    S = (S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d *
        D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] *
        D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]
        )
    O = (O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d *
        D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] *
        D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None,
        :] + D_MODEL_K * D_MODEL_V)
    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL_V + tl.arange(0, BLOCK_MODEL
        ) + offset_s * BLOCK_MODEL + D_MODEL_V
    acc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)
    acc += tl.load(S)
    S += D_MODEL_K * D_MODEL_V
    tl.store(O, acc)
    O += D_MODEL_K * D_MODEL_V
    for i in range(NUM_BLOCK - 2):
        p_v = tl.load(p2)
        S_i = tl.load(S)
        acc = acc * p_v[None, :] + S_i
        tl.store(O, acc)
        p2 += D_MODEL_V
        S += D_MODEL_K * D_MODEL_V
        O += D_MODEL_K * D_MODEL_V
