import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=8), triton.Config({
    'BLOCK_SIZE': 256}, num_warps=8), triton.Config({'BLOCK_SIZE': 128},
    num_warps=8), triton.Config({'BLOCK_SIZE': 64}, num_warps=8), triton.
    Config({'BLOCK_SIZE': 32}, num_warps=8), triton.Config({'BLOCK_SIZE': 
    16}, num_warps=8), triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=4), triton.Config({
    'BLOCK_SIZE': 256}, num_warps=4), triton.Config({'BLOCK_SIZE': 128},
    num_warps=4), triton.Config({'BLOCK_SIZE': 64}, num_warps=4), triton.
    Config({'BLOCK_SIZE': 32}, num_warps=4), triton.Config({'BLOCK_SIZE': 
    16}, num_warps=4)], key=['num_rows', 'num_cols'])
@triton.jit
def layernorm_kernel(a_ptr, batch_stride, row_stride, col_stride, num_rows,
    num_cols, weight_ptr, bias_ptr, eps, out_ptr, BLOCK_SIZE: 'tl.constexpr'):
    """
    IDEA 1: Merge batch and seq len dimension into 1
    IDEA 2: Use tiled row approach
    """
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)
    batch_offset = batch_idx * batch_stride
    row_offset = row_idx * row_stride
    local_sum = 0.0
    for offset in range(0, num_cols, BLOCK_SIZE):
        local_offset = batch_offset + row_offset + offset + tl.arange(0,
            BLOCK_SIZE)
        mask = offset + tl.arange(0, BLOCK_SIZE) < num_cols
        data = tl.load(a_ptr + local_offset, mask=mask, other=0.0)
        local_sum += tl.sum(data)
    mean = local_sum / num_cols
    local_std = 0.0
    for offset in range(0, num_cols, BLOCK_SIZE):
        local_offset = batch_offset + row_offset + offset + tl.arange(0,
            BLOCK_SIZE)
        mask = offset + tl.arange(0, BLOCK_SIZE) < num_cols
        data = tl.load(a_ptr + local_offset, mask=mask, other=mean)
        x = data - mean
        x = x * x
        local_std += tl.sum(x)
    std = local_std / num_cols + eps
    std = tl.sqrt(std)
    for offset in range(0, num_cols, BLOCK_SIZE):
        local_offset = offset + tl.arange(0, BLOCK_SIZE)
        mask = local_offset < num_cols
        w = tl.load(weight_ptr + local_offset, mask=mask, other=0.0)
        b = tl.load(bias_ptr + local_offset, mask=mask, other=0.0)
        local_offset += row_offset + batch_offset
        mask = offset + tl.arange(0, BLOCK_SIZE) < num_cols
        x = tl.load(a_ptr + local_offset, mask=mask, other=0.0)
        norm = w * ((x - mean) / std) + b
        tl.store(out_ptr + local_offset, norm, mask=mask)
