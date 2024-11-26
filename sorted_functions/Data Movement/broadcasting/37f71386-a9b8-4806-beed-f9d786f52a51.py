import triton
import triton.language as tl
import torch

@triton.jit
def one_shot_all_reduce_kernel(buffer_ptrs, signal_pad_ptrs, output_ptr,
    numel: 'tl.constexpr', rank: 'tl.constexpr', world_size: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr', NUMEL_PER_THREAD: 'tl.constexpr'):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size)
    pid = tl.program_id(axis=0)
    buffer_ptrs = buffer_ptrs
    output_ptr = output_ptr
    block_start = pid * BLOCK_SIZE
    while block_start < numel // NUMEL_PER_THREAD:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE
            ) < numel // NUMEL_PER_THREAD
        acc = tl.zeros((BLOCK_SIZE,), tl.uint64)
        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i)
            val = tl.load(buffer_ptr + offsets, mask=mask)
            acc = add_v4_bf16(acc, val)
        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE
