import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(input_ptr, input_batch_stride, input_row_stride,
    output_ptr, num_rows, num_cols, BLOCK_SIZE: 'tl.constexpr'):
    batch_id = tl.program_id(axis=0)
    row_id = tl.program_id(axis=1)
    batch_offset = batch_id * input_batch_stride
    row_offset = row_id * input_row_stride + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < num_cols
    data = tl.load(input_ptr + batch_offset + row_offset, mask, other=-
        float('inf'))
    data = data - tl.max(data, axis=0)
    row_wise_exp = tl.exp(data)
    row_wise_sum = tl.sum(row_wise_exp, axis=0)
    output = row_wise_exp / row_wise_sum
    tl.store(output_ptr + batch_offset + row_offset, output, mask=mask)
