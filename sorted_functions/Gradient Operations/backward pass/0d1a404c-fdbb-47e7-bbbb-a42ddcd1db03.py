import triton
import triton.language as tl
import torch

@triton.jit
def embedding_backward_kernel(grad_output_ptr, grad_weight_ptr, indices_ptr,
    n_elements, embedding_dim: 'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr',
    BLOCK_SIZE_N: 'tl.constexpr'):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim
    grad_output = tl.load(grad_output_ptr + offsets_m[:, None] *
        embedding_dim + offsets_n[None, :], mask=mask_m[:, None] & mask_n[
        None, :], other=0.0)
    grad_weight_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    tl.atomic_add(grad_weight_ptr + grad_weight_offsets, grad_output, mask=
        mask_m[:, None] & mask_n[None, :])
