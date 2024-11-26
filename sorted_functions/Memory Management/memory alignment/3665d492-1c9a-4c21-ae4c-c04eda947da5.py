import triton
import triton.language as tl
import torch

@triton.jit
def barrier_test_kernel(signal_pad_ptrs, RANK: 'tl.constexpr', WORLD_SIZE:
    'tl.constexpr'):
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE)
