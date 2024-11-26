import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_RAGGED': b_r,
    'BLOCK_SIZE_M': b_m}, num_warps=w, num_stages=s) for b_r, b_m, w, s in
    itertools.product(BLOCK_SIZES_RAGGED, BLOCK_SIZES_M, NUM_WARPS,
    NUM_STAGES)], key=['M'])
@triton.jit
def triton_jagged_mean_kernel_variable_length_loop_buffer_then_sum(
    input_ptr_values, input_ptr_offsets, output_ptr, M, BLOCK_SIZE_RAGGED:
    'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    pid_ragged = pid // tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)
    buffer = tl.zeros((BLOCK_SIZE_RAGGED, BLOCK_SIZE_M), dtype=tl.float32)
    block_start_m = pid_m * BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < M
    ragged_start, ragged_end = tl.load(input_ptr_offsets + pid_ragged
        ), tl.load(input_ptr_offsets + (pid_ragged + 1))
    ragged_len = ragged_end - ragged_start
    for block_start_ragged in range(ragged_start, ragged_end, BLOCK_SIZE_RAGGED
        ):
        offsets_ragged = block_start_ragged + tl.arange(0, BLOCK_SIZE_RAGGED)
        mask_ragged = offsets_ragged < ragged_end
        idxs = offsets_ragged[:, None] * M + offsets_m
        mask = mask_ragged[:, None] & mask_m
        buffer += tl.load(input_ptr_values + idxs, mask=mask, other=0)
    buffer_sum = tl.sum(buffer, axis=0)
    buffer_view = buffer_sum.reshape((BLOCK_SIZE_M,))
    buffer_view_mean = buffer_view * (1 / ragged_len)
    output_offsets = offsets_m + pid_ragged * M
    output_mask = output_offsets < M * (pid_ragged + 1)
    tl.store(output_ptr + output_offsets, buffer_view_mean, mask=output_mask)