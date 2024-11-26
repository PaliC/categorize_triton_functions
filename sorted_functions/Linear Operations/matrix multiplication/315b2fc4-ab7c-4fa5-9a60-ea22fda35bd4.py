import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=16), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=4)], key=[])
@triton.jit
def conv2d_kernel(input_ptr, input_batch_stride, input_channel_stride,
    input_row_stride, input_col_stride, height, width, channels, kernel_ptr,
    kernel_height, kernel_width, kernel_dim_stride, kernel_channel_stride,
    kernel_row_stride, kernel_col_stride, bias_ptr, output_ptr,
    output_width, output_batch_stride, output_channel_stride,
    output_row_stride, output_col_stride, BLOCK_SIZE_ROW: 'tl.constexpr',
    BLOCK_SIZE_COL: 'tl.constexpr'):
    batch_idx = tl.program_id(0)
    kernel_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    bias_offset = kernel_idx
    bias = tl.load(bias_ptr + bias_offset)
    batch_offset = batch_idx * input_batch_stride
    output_batch_offset = batch_idx * output_batch_stride
    output_channel_offset = kernel_idx * output_channel_stride
    output_row_offset = row_idx * output_row_stride
    kernel_row_offset = tl.arange(0, BLOCK_SIZE_ROW)
    kernel_row_mask = kernel_row_offset[:, None] < kernel_height
    kernel_row_offset = kernel_row_offset[:, None] * kernel_row_stride
    kernel_col_offset = tl.arange(0, BLOCK_SIZE_COL)
    kernel_col_mask = kernel_col_offset[None, :] < kernel_width
    kernel_col_offset = kernel_col_offset[None, :] * kernel_col_stride
    kernel_mask = kernel_row_mask & kernel_col_mask
    for col_idx in range(output_width):
        elem = 0.0
        input_row_offset = row_idx * kernel_height + tl.arange(0,
            BLOCK_SIZE_ROW)
        input_row_mask = input_row_offset[:, None] < height
        input_row_offset = input_row_offset[:, None] * input_row_stride
        input_col_offset = col_idx * kernel_width + tl.arange(0, BLOCK_SIZE_ROW
            )
        input_col_mask = input_col_offset[None, :] < width
        input_col_offset = input_col_offset[None, :] * input_col_stride
        input_mask = input_row_mask & input_col_mask
        for c in range(channels):
            input_offset = (input_ptr + batch_offset + c *
                input_channel_stride + input_row_offset + input_col_offset)
            input_data = tl.load(input_offset, input_mask)
            kernel_offset = (kernel_ptr + kernel_idx * kernel_dim_stride + 
                c * kernel_channel_stride + kernel_row_offset +
                kernel_col_offset)
            kernel_data = tl.load(kernel_offset, kernel_mask)
            dot_prdct = input_data * kernel_data
            elem += tl.sum(dot_prdct)
        output_offset = (output_ptr + output_batch_offset +
            output_channel_offset + output_row_offset + col_idx)
        tl.store(output_offset, elem + bias)
