import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_N': 128,
    'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4), triton.Config({
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
    num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 64}, num_stages=5, num_warps=2), triton.Config({
    'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 64,
    'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
    num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_N': 64,
    'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2)], key=['K', 'N'])
@triton.jit
def dequantize_kernel(b_ptr, b_scale_ptr, fpb_ptr, K, N, stride_bk,
    stride_bn, stride_fpbk, stride_fpbn, BLOCK_SIZE_N: 'tl.constexpr',
    BLOCK_SIZE_K: 'tl.constexpr'):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + (
        n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + (
        n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
    bs_offs = n_block_idx * BLOCK_SIZE_N + offs_n[None, :]
    n_mask = n_block_idx * BLOCK_SIZE_N + offs_n[None, :] < N
    mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None] < K) & n_mask
    int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
    tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
