import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_rms_kernel(out_ptr_base, stride_out_row, in_ptr_base, stride_x_row,
    stride_x_col, weight_ptr, num_rows: 'tl.constexpr', num_cols:
    'tl.constexpr', block_size: 'tl.constexpr'):
    row_index = tl.program_id(0)
    in_ptr_row = in_ptr_base + row_index * stride_x_row
    out_ptr_row = out_ptr_base + row_index * stride_out_row
    in_block_ptr = tl.make_block_ptr(base=in_ptr_base, shape=(num_rows,
        num_cols), strides=(stride_x_row, stride_x_col), offsets=(row_index,
        0), block_shape=(1, block_size), order=(1, 0))
    variance_row = 0.0
    eps = 1e-08
    test = tl.zeros([5], dtype=tl.float32)
    summer = 0.0
    variance = 0.0
    for col_index in range(0, block_size // num_cols):
        col_block = tl.load(in_block_ptr[0], boundary_check=(0, 1))
        variance += tl.sum(col_block * col_block, axis=0)
        in_block_ptr = tl.advance(in_block_ptr, (0, block_size))
        col_offsets = col_index + tl.arange(0, block_size)
        col_mask = col_offsets < num_cols
        col_block2 = tl.load(in_ptr_row + col_offsets, mask=col_mask, other=0.0
            )
        variance_row += tl.sum(col_block2 * col_block2, axis=0)
    tl.device_print('summer, variance: ', summer, variance)
    variance /= num_cols
    rstdev = 1 / tl.sqrt(variance + eps)
    for start_col in range(0, num_cols, block_size):
        col_offsets = start_col + tl.arange(0, block_size)
        col_mask = col_offsets < num_cols
        weights = tl.load(weight_ptr + col_offsets, mask=col_mask)
        in_block = tl.load(in_ptr_row + col_offsets, mask=col_mask, other=
            0.0, eviction_policy='evict_first')
        col_block_rms = in_block * rstdev
        out = weights * col_block_rms
        tl.store(out_ptr_row + col_offsets, out, mask=col_mask)
