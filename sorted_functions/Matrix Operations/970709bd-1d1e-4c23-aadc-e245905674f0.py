import triton
import triton.language as tl
import torch

@triton.jit
def gemm_split_k_kernel(a_ptr, b_ptr, c_ptr, stride_am, stride_ak,
    stride_bk, stride_bn, stride_cm, stride_cn, scale_a, scale_b, m, n, k,
    block_m: 'tl.constexpr', block_n: 'tl.constexpr', block_k:
    'tl.constexpr', split_k: 'tl.constexpr', group_m: 'tl.constexpr'):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k * split_k)
    pid_m, pid_n = grouped_launch(pid, m, n, block_m, block_n, group_m)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
        stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] *
        stride_bn)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, grid_k):
        k_remaining = k - k_ * (block_k * split_k)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk
    acc = scale_a * scale_b * acc
    acc
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        )
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]
    tl.atomic_add(c_ptrs, acc, mask=mask)
