import triton
import triton.language as tl
import torch

@triton.jit
def k_kernel_per_block_int8(X, X_int8, BLK: 'tl.constexpr', Scale, L, C:
    'tl.constexpr', scale_stride):
    off_b = tl.program_id(1)
    off_blk = tl.program_id(0)
    x_offset = off_b * L * C
    offs_m = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, 128)
    x_ptrs = X + x_offset + offs_m[:, None] * C + offs_k[None, :]
    x_int8_ptrs = X_int8 + x_offset + offs_m[:, None] * C + offs_k[None, :]
    scale_ptrs = Scale + off_b * scale_stride + off_blk
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < L) & (tl.arange(0, 128) < 
        96)[None, :])
    scale = tl.max(tl.abs(x)) / 127.0
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8
    tl.store(x_int8_ptrs, x_int8, mask=(offs_m[:, None] < L) & (tl.arange(0,
        128) < 96)[None, :])
    tl.store(scale_ptrs, scale)
