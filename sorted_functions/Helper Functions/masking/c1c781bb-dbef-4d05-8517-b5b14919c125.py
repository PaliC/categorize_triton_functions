import triton
import triton.language as tl
import torch

@triton.jit
def recompute_mask_kernel(mask, B, H, M, N, dropout_p, seed, offset):
    row, b, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_base = b * H * M * N + h * M * N + row * N
    BLOCK: 'tl.constexpr' = 1024
    offs_base += tl.arange(0, BLOCK)
    for start_n in range(0, N, BLOCK):
        offs = start_n + offs_base
        rng_offs = offset + offs
        pmask = tl.rand(seed, rng_offs, n_rounds=6) > dropout_p
        row_mask = start_n + tl.arange(0, BLOCK) < N
        tl.store(mask + offs, pmask, mask=row_mask)
