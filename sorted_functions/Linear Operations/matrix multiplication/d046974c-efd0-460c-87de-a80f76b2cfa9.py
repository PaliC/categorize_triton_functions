import triton
import triton.language as tl
import torch

@triton.jit
def dot(BLOCK_M: 'tl.constexpr', QDIM: 'tl.constexpr', KDIM: 'tl.constexpr',
    q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)
