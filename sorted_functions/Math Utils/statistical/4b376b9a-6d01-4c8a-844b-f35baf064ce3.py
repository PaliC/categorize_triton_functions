import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_N': b_n,
    'BLOCK_SIZE_K': b_k}, num_warps=w) for b_n, b_k, w in itertools.product
    ([(4 ** n) for n in range(6)], [(4 ** n) for n in range(4)], [2, 4, 8])
    ], key=['N'])
@triton.jit
def triton_sum_kernel_2D_result_dim_1_sum_then_buffer(input_ptr, output_ptr,
    M: 'tl.constexpr', N: 'tl.constexpr', K: 'tl.constexpr', BLOCK_SIZE_N:
    'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'):
    """
    Modification to triton_sum_kernel_2D_result_dim_1() which uses a buffer to store intermediate results,
    enabling reducing over a large middle dimension for 3D input tensors
    """
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_k = pid % tl.cdiv(K, BLOCK_SIZE_K)
    buffer = tl.zeros((1, BLOCK_SIZE_K), dtype=tl.float32)
    block_start_k = pid_k * BLOCK_SIZE_K
    offsets_k = block_start_k + tl.arange(0, BLOCK_SIZE_K)
    mask_k = offsets_k < K
    for block_start_n in range(0, N, BLOCK_SIZE_N):
        offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offsets_n < N
        idxs_base = offsets_n[:, None] * K + offsets_k
        idxs = idxs_base + pid_m * N * K
        mask = mask_n[:, None] & mask_k
        input = tl.load(input_ptr + idxs, mask=mask, other=0)
        buffer += tl.sum(input, axis=0)
    buffer_view = buffer.reshape((BLOCK_SIZE_K,))
    output_offsets = pid_m * K + offsets_k
    tl.store(output_ptr + output_offsets, buffer_view, mask=mask_k)
