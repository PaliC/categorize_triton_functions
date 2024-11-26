import triton
import triton.language as tl
import torch

@triton.jit
def get_sample_randn(pid, step, n_rays, n_steps, BLOCK_SIZE, seed_buffer):
    offs = pid * BLOCK_SIZE * n_steps + 1
    i1 = offs + step + tl.arange(0, BLOCK_SIZE) * n_steps
    i2 = n_rays * n_steps + i1
    return int_to_randn(i1, i2, seed_buffer)
