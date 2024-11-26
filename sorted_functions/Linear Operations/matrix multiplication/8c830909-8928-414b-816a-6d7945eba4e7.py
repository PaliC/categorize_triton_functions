import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bn,
    stride_bk, stride_cn, stride_cm):
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.arange(0, BLOCK_SIZE)
    a_ptrs = A + m * stride_am + k * stride_ak
    b_ptrs = B + k * stride_bk + n * stride_bn
    c_ptrs = C + m * stride_cm + n * stride_cn
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.dot(a, b)
    tl.atomic_add(c_ptrs, c)
