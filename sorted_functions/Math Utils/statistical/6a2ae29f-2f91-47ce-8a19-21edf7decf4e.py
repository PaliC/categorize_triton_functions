import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_K': b}, num_warps=w) for
    b, w in itertools.product([2, 4, 16, 32, 128, 256], [2, 4, 8])], key=['N'])
@triton.jit
def triton_sum_kernel_2D_result_dim_1(input_ptr, output_ptr, M:
    'tl.constexpr', N: 'tl.constexpr', K: 'tl.constexpr', BLOCK_SIZE_N:
    'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_k = pid % tl.cdiv(K, BLOCK_SIZE_K)
    block_start_n = 0
    block_start_k = pid_k * BLOCK_SIZE_K
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)
    mask_n = offsets_n < N
    mask_k = offsets_k < K
    idxs_base = offsets_n[:, None] * K + offsets_k
    idxs = idxs_base + pid_m * N * K
    mask = mask_n[:, None] & mask_k
    input = tl.load(input_ptr + idxs, mask=mask, other=0)
    output = tl.sum(input, axis=0)
    output_offsets = pid_m * K + offsets_k
    tl.store(output_ptr + output_offsets, output, mask=mask_k)
