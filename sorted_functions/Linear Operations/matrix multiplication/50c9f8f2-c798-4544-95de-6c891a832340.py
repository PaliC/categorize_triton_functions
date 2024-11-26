import triton
import triton.language as tl
import torch

@triton.jit
def bmm_kernel(x_ptr, x_stride_b, x_stride_m, x_stride_k, y_ptr, y_stride_b,
    y_stride_k, y_stride_n, o_ptr, o_stride_b, o_stride_m, o_stride_n, m:
    'tl.constexpr', n: 'tl.constexpr', k: 'tl.constexpr', block_size_m:
    'tl.constexpr', block_size_n: 'tl.constexpr', block_size_k:
    'tl.constexpr', group_size_m: 'tl.constexpr'):
    batch_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)
    num_blocks_m = tl.cdiv(m, block_size_m)
    num_blocks_n = tl.cdiv(n, block_size_n)
    num_blocks_g = group_size_m * num_blocks_n
    group_idx = block_idx // num_blocks_g
    first_block_m = group_idx * group_size_m
    group_size_m_ = min(num_blocks_m - first_block_m, group_size_m)
    block_idx_m = first_block_m + block_idx % group_size_m_
    block_idx_n = block_idx % num_blocks_g // group_size_m_
    block_ptrs_m = (block_idx_m * block_size_m + tl.arange(0, block_size_m)
        ) % m
    block_ptrs_n = (block_idx_n * block_size_n + tl.arange(0, block_size_n)
        ) % n
    offsets_k = tl.arange(0, block_size_k)
    x_block_ptrs = x_ptr + (block_ptrs_m[:, None] * x_stride_m + offsets_k[
        None, :] * x_stride_k) + batch_idx * x_stride_b
    y_block_ptrs = y_ptr + (offsets_k[:, None] * y_stride_k + block_ptrs_n[
        None, :] * y_stride_n) + batch_idx * y_stride_b
    o = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for block_idx_k in range(0, tl.cdiv(k, block_size_k)):
        block_offs_k = block_idx_k * block_size_k + offsets_k
        x = tl.load(x_block_ptrs, mask=block_offs_k[None, :] < k, other=0.0)
        y = tl.load(y_block_ptrs, mask=block_offs_k[:, None] < k, other=0.0)
        o += tl.dot(x, y)
        x_block_ptrs += block_size_k * x_stride_k
        y_block_ptrs += block_size_k * y_stride_k
    block_ptrs_m_ = block_idx_m * block_size_m + tl.arange(0, block_size_m)[
        :, None]
    block_ptrs_n_ = block_idx_n * block_size_n + tl.arange(0, block_size_n)[
        None, :]
    o_ptrs = o_ptr + (block_ptrs_m_ * o_stride_m + block_ptrs_n_ * o_stride_n
        ) + batch_idx * o_stride_b
    block_mask_mn = (block_ptrs_m_ < m) & (block_ptrs_n_ < n)
    tl.store(o_ptrs, o, mask=block_mask_mn)
