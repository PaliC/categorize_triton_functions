import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_NON_REDUCE_DIM': b_nr,
    'BLOCK_SIZE_REDUCE_DIM': b_r}, num_warps=w) for b_nr, b_r, w in
    itertools.product([2, 4, 8, 16], [2, 4, 8, 16], [2, 4, 8])], key=['M', 'N']
    )
@triton.jit
def triton_sum_kernel_1D_result_sum_then_buffer(input_ptr, output_ptr, M, N,
    BLOCK_SIZE_NON_REDUCE_DIM: 'tl.constexpr', BLOCK_SIZE_REDUCE_DIM:
    'tl.constexpr', dim: 'tl.constexpr'):
    """
    Sum blocks of input using Triton and store in buffer
    """
    pid = tl.program_id(axis=0)
    reduce_dim_len = M if dim == 0 else N
    non_reduce_dim_len = N if dim == 0 else M
    buffer = tl.zeros((1, BLOCK_SIZE_NON_REDUCE_DIM), dtype=tl.float32)
    block_start_non_reduce_dim = pid * BLOCK_SIZE_NON_REDUCE_DIM
    offsets_non_reduce_dim = block_start_non_reduce_dim + tl.arange(0,
        BLOCK_SIZE_NON_REDUCE_DIM)
    mask_non_reduce_dim = offsets_non_reduce_dim < non_reduce_dim_len
    for block_start_reduce_dim in range(0, reduce_dim_len,
        BLOCK_SIZE_REDUCE_DIM):
        offsets_reduce_dim = block_start_reduce_dim + tl.arange(0,
            BLOCK_SIZE_REDUCE_DIM)
        mask_reduce_dim = offsets_reduce_dim < reduce_dim_len
        idxs, mask = None, None
        if dim == 0:
            idxs = offsets_reduce_dim[:, None
                ] * non_reduce_dim_len + offsets_non_reduce_dim
            mask = mask_reduce_dim[:, None] & mask_non_reduce_dim
        elif dim == 1:
            idxs = offsets_non_reduce_dim[:, None
                ] * reduce_dim_len + offsets_reduce_dim
            mask = mask_non_reduce_dim[:, None] & mask_reduce_dim
        input = tl.load(input_ptr + idxs, mask=mask, other=mask)
        buffer += tl.sum(input, axis=dim)
    buffer_view = buffer.reshape((BLOCK_SIZE_NON_REDUCE_DIM,))
    tl.store(output_ptr + offsets_non_reduce_dim, buffer_view, mask=
        mask_non_reduce_dim)