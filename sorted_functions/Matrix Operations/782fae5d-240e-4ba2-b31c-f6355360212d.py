import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel_with_block_pointers(a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K:
    'tl.constexpr', GROUP_M: 'tl.constexpr'):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % GROUP_M
    pid_n = pid % num_pid_in_group // GROUP_M
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(
        stride_am, stride_ak), offsets=(pid_m * BLOCK_M, 0), block_shape=(
        BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(
        stride_bk, stride_bn), offsets=(0, pid_n * BLOCK_N), block_shape=(
        BLOCK_K, BLOCK_N), order=(1, 0))
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    c = accumulator
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(
        stride_cm, stride_cn), offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))
