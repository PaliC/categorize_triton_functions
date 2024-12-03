import triton
import triton.language as tl
import torch

@triton.jit
def _rmsnorm_kernel_fwd(x_ptr, w_ptr, z_ptr, K, eps, BLOCK_SIZE:
    'tl.constexpr'=8):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * K
    w_row_ptr = w_ptr + row_idx * K
    z_row_ptr = z_ptr + row_idx * K
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_index in range(0, K, BLOCK_SIZE):
        col_offsets = col_index + tl.arange(0, BLOCK_SIZE)
        x_ptrs = x_row_ptr + col_offsets
        x = tl.load(x_ptrs, mask=col_offsets < K, other=0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / K
    rsqrt = 1 / tl.sqrt(var + eps)
    for col_index in range(0, K, BLOCK_SIZE):
        col_offsets = col_index + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < K
        x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
        w = tl.load(w_ptr + col_offsets, mask=mask)
        normed = x * rsqrt
        normed = normed
        z = normed * w
        tl.store(z_row_ptr + col_offsets, z, mask=mask)
