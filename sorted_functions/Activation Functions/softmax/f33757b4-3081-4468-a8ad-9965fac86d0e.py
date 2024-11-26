import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride,
    output_row_stride, n_cols, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    input_row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
    output_row_ptr = output_ptr + row_idx * output_row_stride + col_offsets
    logits = tl.load(input_row_ptr, mask=mask, other=float('-inf'))
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0) + 1e-06
    softmax_output = exp_logits / sum_exp_logits
    tl.store(output_row_ptr, softmax_output, mask=mask)
