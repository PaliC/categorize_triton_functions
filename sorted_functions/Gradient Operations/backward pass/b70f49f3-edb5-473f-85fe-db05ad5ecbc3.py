import triton
import triton.language as tl
import torch

@triton.jit
def joint_second_order_bwd(coord_ptr: 'tl.tensor', coord_grad_ptr:
    'tl.tensor', sph_grad_ptr: 'tl.tensor', block_size: 'tl.constexpr',
    coord_numel: 'tl.constexpr', output_numel: 'tl.constexpr'):
    block_id = tl.program_id(0)
    coord_stride = 3
    coord_striding = tl.arange(0, block_size) * coord_stride
    coord_row_offset = coord_striding + block_size * coord_stride * block_id
    x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset <
        coord_numel)
    y = tl.load(coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 <
        coord_numel)
    z = tl.load(coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 <
        coord_numel)
    output_stride = 9
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + block_size * output_stride * block_id
    CONST_00 = 3.87298334620742
    CONST_01 = 2.23606797749979
    CONST_02 = 4.47213595499958
    CONST_03 = tl.sqrt(3.0)
    g_Y10 = tl.load(sph_grad_ptr + output_row_offset + 1, mask=
        output_row_offset + 1 < output_numel)
    g_Y11 = tl.load(sph_grad_ptr + output_row_offset + 2, mask=
        output_row_offset + 2 < output_numel)
    g_Y12 = tl.load(sph_grad_ptr + output_row_offset + 3, mask=
        output_row_offset + 3 < output_numel)
    g_Y20 = tl.load(sph_grad_ptr + output_row_offset + 4, mask=
        output_row_offset + 4 < output_numel)
    g_Y21 = tl.load(sph_grad_ptr + output_row_offset + 5, mask=
        output_row_offset + 5 < output_numel)
    g_Y22 = tl.load(sph_grad_ptr + output_row_offset + 6, mask=
        output_row_offset + 6 < output_numel)
    g_Y23 = tl.load(sph_grad_ptr + output_row_offset + 7, mask=
        output_row_offset + 7 < output_numel)
    g_Y24 = tl.load(sph_grad_ptr + output_row_offset + 8, mask=
        output_row_offset + 8 < output_numel)
    g_x = (CONST_00 * g_Y20 * z + CONST_00 * g_Y21 * y - CONST_01 * g_Y22 *
        x - CONST_00 * g_Y24 * x + CONST_03 * g_Y10)
    g_y = (CONST_00 * g_Y21 * x + CONST_02 * g_Y22 * y + CONST_00 * g_Y23 *
        z + CONST_03 * g_Y11)
    g_z = (CONST_00 * g_Y20 * x - CONST_01 * g_Y22 * z + CONST_00 * g_Y23 *
        y + CONST_00 * g_Y24 * z + CONST_03 * g_Y12)
    tl.store(coord_grad_ptr + coord_row_offset, g_x, mask=coord_row_offset <
        coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 1, g_y, mask=
        coord_row_offset + 1 < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 2, g_z, mask=
        coord_row_offset + 2 < coord_numel)
