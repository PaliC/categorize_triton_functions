import triton
import triton.language as tl
import torch

@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K: 'tl.constexpr', stride_am,
    stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K:
    'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr'):
    tl.static_assert(K % (4 * BLOCK_SIZE_K) == 0,
        'K / 4 must be divisible by BLOCK_SIZE_K => K divisible by 4*BLOCK_SIZE_K'
        )
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    """
        This part of the code generates pointers to the specific blocks of matrices A and B that the current thread block will process.

        As described in the PyTorch documentation, a stride refers to the step size needed to move from one element to the next along a given dimension:

        For matrix A: stride_am = A.stride(0) = K (stride along the rows), and stride_ak = A.stride(1) = 1 (stride along the columns).
        For matrix B: stride_bk = B.stride(0) = N (stride along the rows), and stride_bn = B.stride(1) = 1 (stride along the columns).
        Now, let's break down the pointer generation:

        offs_am[:, None] creates a column of shape [BLOCK_SIZE_M, 1], which represents the row indices of matrix A that this block is processing. It is multiplied by K (the number of columns in matrix A) since A is stored in row-major order. So, the element at position (i, j) in A is located at index i*K + j in memory.
        offs_k[None, BLOCK_SIZE_K] creates a row vector representing the column indices of the block, i.e., a range from 0 to BLOCK_SIZE_K. This is used to compute the positions of the columns within the block.
        When combined, the result has the shape [BLOCK_SIZE_M, BLOCK_SIZE_K], where each entry (i, j) points to the element in matrix A at position (i, j) for the current block.

        The same logic is applied to matrix B, but the resulting shape is [BLOCK_SIZE_K, BLOCK_SIZE_N], representing the block of matrix B that the thread block will work on.
    """
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
        stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] *
        stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    """
        We split the loop into two layers. The outer loop runs 4 times, and each iteration focuses on a specific portion of matrix A.

        For example, when i = 0, we’re only concerned with the blocks of matrix A that cover the range from 0 to K // (4 * BLOCK_SIZE_K).
        Since matrix B is packed, its first dimension is effectively divided by 4. So, while we process the first segment of matrix A,
        we still iterate over the entire first dimension of matrix B.

        In each of the 4 iterations of the outer loop, we go through the full blocks of matrix B, but what changes is the data we extract.
        Matrix B elements contain 4 weights, all packed into an int8 format, and during each iteration of the outer loop,
        we extract a different weight by using bitwise shifting operations. This way, we access a unique weight on each pass.
    """
    for i in range(4):
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] *
            stride_bn)
        for j in range(0, tl.cdiv(K // 4, BLOCK_SIZE_K)):
            k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                other=0)
            b_uint8 = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
            mask = 3 << 2 * i
            b = (b_uint8 & mask) >> 2 * i
            tensor_full = tl.full((1,), 1, dtype=tl.int8)
            accumulator += tl.dot(a, b - tensor_full, out_dtype=tl.int32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :
        ]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
