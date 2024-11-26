import triton
import triton.language as tl
import torch

@triton.jit
def joint_second_order_fwd(coord_ptr: 'tl.tensor', output_ptr: 'tl.tensor',
    block_size: 'tl.constexpr', coord_numel: 'tl.constexpr', output_numel:
    'tl.constexpr'):
    """
    This Triton implementation includes l=0, 1, 2 within the
    same kernel, as it would be a common operation.
    """
    coord_stride = 3
    block_id = tl.program_id(0)
    coord_striding = tl.arange(0, block_size) * coord_stride
    coord_row_offset = coord_striding + block_size * coord_stride * block_id
    x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset <
        coord_numel)
    y = tl.load(coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 <
        coord_numel)
    z = tl.load(coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 <
        coord_numel)
    CONST_00 = 3.87298334620742
    CONST_01 = 2.23606797749979
    CONST_02 = -1.11803398874989
    CONST_03 = 1.93649167310371
    CONST_04 = tl.sqrt(3.0)
    Y10 = CONST_04 * x
    Y11 = CONST_04 * y
    Y12 = CONST_04 * z
    Y20 = CONST_00 * x * z
    Y21 = CONST_00 * x * y
    Y23 = CONST_00 * y * z
    Y22 = CONST_02 * x * x + CONST_01 * y * y + CONST_02 * z * z
    Y24 = -CONST_03 * x * x + CONST_03 * z * z
    output_stride = 9
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + block_size * output_stride * block_id
    tl.store(output_ptr + output_row_offset, 1.0, mask=output_row_offset <
        output_numel)
    tl.store(output_ptr + output_row_offset + 1, Y10, mask=
        output_row_offset + 1 < output_numel)
    tl.store(output_ptr + output_row_offset + 2, Y11, mask=
        output_row_offset + 2 < output_numel)
    tl.store(output_ptr + output_row_offset + 3, Y12, mask=
        output_row_offset + 3 < output_numel)
    tl.store(output_ptr + output_row_offset + 4, Y20, mask=
        output_row_offset + 4 < output_numel)
    tl.store(output_ptr + output_row_offset + 5, Y21, mask=
        output_row_offset + 5 < output_numel)
    tl.store(output_ptr + output_row_offset + 6, Y22, mask=
        output_row_offset + 6 < output_numel)
    tl.store(output_ptr + output_row_offset + 7, Y23, mask=
        output_row_offset + 6 < output_numel)
    tl.store(output_ptr + output_row_offset + 8, Y24, mask=
        output_row_offset + 7 < output_numel)
