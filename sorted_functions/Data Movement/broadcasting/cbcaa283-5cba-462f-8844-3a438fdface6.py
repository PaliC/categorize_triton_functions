import triton
import triton.language as tl
import torch

@triton.jit
def one_shot_reduce_scatter_kernel(buffer_ptrs, signal_pad_ptrs, output_ptr,
    numel: 'tl.constexpr', rank: 'tl.constexpr', world_size: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr', NUMEL_PER_THREAD: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    per_rank_numel = numel // world_size
    buffer_ptrs = buffer_ptrs
    output_ptr = output_ptr
    block_start = pid * BLOCK_SIZE
    while block_start < per_rank_numel // NUMEL_PER_THREAD:
        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE
            ) < per_rank_numel // NUMEL_PER_THREAD
        acc_hi = tl.zeros((BLOCK_SIZE,), tl.uint64)
        acc_lo = tl.zeros((BLOCK_SIZE,), tl.uint64)
        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i
                ) + rank * per_rank_numel // NUMEL_PER_THREAD * 2
            hi, lo = load_128(buffer_ptr + offsets, mask=mask)
            acc_hi, acc_lo = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        tl.store(output_ptr + offsets + 0, acc_hi, mask=mask)
        tl.store(output_ptr + offsets + 1, acc_lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE
