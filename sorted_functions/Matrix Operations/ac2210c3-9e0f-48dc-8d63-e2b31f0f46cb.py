import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk,
    stride_bn, stride_cm, stride_cn, c_ptr, BLOCK_SIZE_M: 'tl.constexpr',
    BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr',
    GROUP_SIZE_M: 'tl.constexpr', K_EXACTLY_DIVISIBLE_BY_BLOCK: 'tl.constexpr'
    ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
        stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] *
        stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_remaining in range(K, 0, -BLOCK_SIZE_K):
        if K_EXACTLY_DIVISIBLE_BY_BLOCK:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            mask = tl.arange(0, BLOCK_SIZE_K) < k_remaining
            a = tl.load(a_ptrs, mask=mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask[:, None], other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :
        ]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)