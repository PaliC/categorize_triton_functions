import triton
import triton.language as tl
import torch

@triton.jit
def jagged_reduce_sum(seq_offsets, Jagged, Out, D, stride_jn, stride_ob,
    BLOCK_D: 'tl.constexpr'):
    """
    Computing Out = Jagged + Dense
    JaggedA has shape (sum_B(N_i), D), Dense has shape (B, D), and Out has shape (sum_B(N_i), D)
    """
    off_b = tl.program_id(0)
    off_d = tl.program_id(1) * BLOCK_D
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    Jagged += seq_start * stride_jn
    Out += off_b * stride_ob
    offs_d = off_d + tl.arange(0, BLOCK_D)
    jagged_ptrs = Jagged + offs_d
    out_ptrs = Out + offs_d
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for _ in range(0, seq_len):
        jg = tl.load(jagged_ptrs, mask=offs_d < D)
        accumulator += jg
        jagged_ptrs += stride_jn
    out = accumulator
    tl.store(out_ptrs, out, mask=offs_d < D)
