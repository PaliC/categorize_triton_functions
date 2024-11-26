import triton
import triton.language as tl
import torch

@triton.jit
def one_shot_all_gather_kernel(buffer_ptrs, signal_pad_ptrs, output_ptr,
    numel: 'tl.constexpr', rank: 'tl.constexpr', world_size: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr', NUMEL_PER_THREAD: 'tl.constexpr'):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size)
    pid = tl.program_id(axis=0)
    buffer_ptrs = buffer_ptrs
    output_ptr = output_ptr
    block_start = pid * BLOCK_SIZE
    while block_start < numel // NUMEL_PER_THREAD:
        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE
            ) < numel // NUMEL_PER_THREAD
        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i)
            hi, lo = load_128(buffer_ptr + offsets, mask=mask)
            scale_factor_for_uint64_ptr = (tl.uint64.primitive_bitwidth //
                tl.bfloat16.primitive_bitwidth)
            tl.store(output_ptr + i * numel // scale_factor_for_uint64_ptr +
                offsets + 0, hi, mask=mask)
            tl.store(output_ptr + i * numel // scale_factor_for_uint64_ptr +
                offsets + 1, lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE
