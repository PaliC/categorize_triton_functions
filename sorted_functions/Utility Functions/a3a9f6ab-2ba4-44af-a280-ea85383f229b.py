import triton
import triton.language as tl
import torch

@triton.jit
def _int_to_randn_kernel(x1, x2, out, N: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr', seed: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < N
    x1_buffer = tl.load(x1 + offs, mask=offs_mask)
    x2_buffer = tl.load(x2 + offs, mask=offs_mask)
    seed_buffer = tl.full((BLOCK_SIZE,), seed, dtype=tl.int64)
    r = _int_to_randn(x1_buffer, x2_buffer, seed_buffer)
    tl.store(out + offs, r, mask=offs_mask)
