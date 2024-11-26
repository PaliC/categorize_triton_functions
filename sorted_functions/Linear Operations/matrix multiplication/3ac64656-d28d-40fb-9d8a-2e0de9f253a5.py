import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N':
    32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=2), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages
    =5, num_warps=2), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 64}, num_stages=6, num_warps=2), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128},
    num_stages=4, num_warps=2), triton.Config({'BLOCK_SIZE_M': 32,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 
    128}, num_stages=6, num_warps=2)], key=['M', 'N', 'K', 'PK'])
@triton.jit
def _matmul_partition_k(a_ptr, b_ptr, c_buf_ptr, M, N, K, PK, PK_SIZE,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cb_m, stride_cb_n,
    stride_cb_k, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr',
    BLOCK_SIZE_K: 'tl.constexpr'):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_pk = tl.program_id(axis=2)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = (pid_pk * PK_SIZE + tl.arange(0, BLOCK_SIZE_K)) % K
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
        stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] *
        stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(PK_SIZE, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_ck = pid_pk
    c_buf_ptrs = c_buf_ptr + stride_cb_m * offs_cm[:, None, None
        ] + stride_cb_n * offs_cn[None, :, None] + stride_cb_k * offs_ck[
        None, None, :]
    tl.store(c_buf_ptrs, acc[:, :, None])
