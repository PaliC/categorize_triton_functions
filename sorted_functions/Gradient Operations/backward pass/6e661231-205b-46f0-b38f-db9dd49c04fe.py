import triton
import triton.language as tl
import torch

@triton.jit
def third_order_bwd(coord_ptr: 'tl.tensor', coord_grad_ptr: 'tl.tensor',
    sph_grad_ptr: 'tl.tensor', block_size: 'tl.constexpr', coord_numel:
    'tl.constexpr', output_numel: 'tl.constexpr', col_offset:
    'tl.constexpr', output_stride: 'tl.constexpr'):
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
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = (output_striding + block_size * output_stride *
        block_id + col_offset)
    g_0 = tl.load(sph_grad_ptr + output_row_offset, mask=output_row_offset <
        output_numel)
    g_1 = tl.load(sph_grad_ptr + output_row_offset + 1, mask=
        output_row_offset + 1 < output_numel)
    g_2 = tl.load(sph_grad_ptr + output_row_offset + 2, mask=
        output_row_offset + 2 < output_numel)
    g_3 = tl.load(sph_grad_ptr + output_row_offset + 3, mask=
        output_row_offset + 3 < output_numel)
    g_4 = tl.load(sph_grad_ptr + output_row_offset + 4, mask=
        output_row_offset + 4 < output_numel)
    g_5 = tl.load(sph_grad_ptr + output_row_offset + 5, mask=
        output_row_offset + 5 < output_numel)
    g_6 = tl.load(sph_grad_ptr + output_row_offset + 6, mask=
        output_row_offset + 6 < output_numel)
    CONST002 = 6.48074069840786
    CONST005 = 12.9614813968157
    CONST007 = -3.96862696659689
    CONST008 = -12.5499003980111
    CONST009 = -10.2469507659596
    CONST010 = -7.93725393319377
    CONST011 = -6.27495019900557
    CONST012 = -5.1234753829798
    CONST013 = -4.8605555238059
    CONST014 = -3.24037034920393
    CONST015 = -1.62018517460197
    VAR08 = x * x
    VAR17 = y * y
    VAR26 = z * z
    g_x = tl.load(coord_grad_ptr + coord_row_offset, mask=coord_row_offset <
        coord_numel)
    g_y = tl.load(coord_grad_ptr + coord_row_offset + 1, mask=
        coord_row_offset + 1 < coord_numel)
    g_z = tl.load(coord_grad_ptr + coord_row_offset + 2, mask=
        coord_row_offset + 2 < coord_numel)
    g_x += (CONST008 * g_6 * x * z - CONST009 * g_1 * y * z + CONST009 *
        g_5 * x * y + CONST010 * g_3 * x * y + CONST014 * g_4 * x * z + g_0 *
        (CONST011 * VAR08 - CONST011 * VAR26) + g_2 * (CONST002 * VAR17 + 
        CONST013 * VAR08 + CONST015 * VAR26))
    g_y += (CONST005 * g_2 * x * y + CONST005 * g_4 * y * z - CONST009 *
        g_1 * x * z + g_3 * (CONST007 * VAR08 + CONST007 * VAR26 - CONST010 *
        VAR17) + g_5 * (CONST012 * VAR08 - CONST012 * VAR26))
    g_z += (-CONST008 * g_0 * x * z - CONST009 * g_1 * x * y - CONST009 *
        g_5 * y * z + CONST010 * g_3 * y * z + CONST014 * g_2 * x * z + g_4 *
        (CONST002 * VAR17 + CONST013 * VAR26 + CONST015 * VAR08) + g_6 * (
        CONST011 * VAR08 - CONST011 * VAR26))
    tl.store(coord_grad_ptr + coord_row_offset, g_x, mask=coord_row_offset <
        coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 1, g_y, mask=
        coord_row_offset + 1 < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 2, g_z, mask=
        coord_row_offset + 2 < coord_numel)
