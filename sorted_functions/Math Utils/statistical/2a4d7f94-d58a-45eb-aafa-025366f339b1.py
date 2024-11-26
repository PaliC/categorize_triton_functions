import triton
import triton.language as tl
import torch

@triton.jit
def fourth_order_fwd(coord_ptr: 'tl.tensor', output_ptr: 'tl.tensor',
    block_size: 'tl.constexpr', coord_numel: 'tl.constexpr', output_numel:
    'tl.constexpr', col_offset: 'tl.constexpr', output_stride: 'tl.constexpr'):
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
    CONST000 = 1.125
    CONST001 = 2.25
    CONST002 = 3.0
    CONST005 = 2.21852991866236
    CONST007 = 9.48683298050514
    CONST010 = 20.1246117974981
    CONST011 = -18.8248505970167
    CONST012 = -13.3111795119741
    CONST013 = -10.0623058987491
    CONST014 = -9.0
    CONST015 = -8.87411967464942
    CONST016 = -7.11512473537885
    CONST017 = -6.27495019900557
    CONST018 = -3.35410196624968
    CONST019 = -1.67705098312484
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    Y00 = CONST015 * VAR07 * z - CONST015 * VAR25 * x
    Y01 = y * (-CONST011 * VAR26 * x + CONST017 * VAR07)
    Y02 = CONST018 * VAR07 * z + x * (CONST010 * VAR17 * z + CONST018 * VAR25)
    Y03 = CONST016 * VAR07 * y + x * (CONST007 * VAR16 + CONST016 * VAR26 * y)
    Y04 = (CONST000 * VAR06 + CONST000 * VAR24 + CONST002 * VAR15 + 
        CONST014 * VAR17 * VAR26 + VAR08 * (CONST001 * VAR26 + CONST014 *
        VAR17))
    Y05 = CONST016 * VAR25 * y + z * (CONST007 * VAR16 + CONST016 * VAR08 * y)
    Y06 = -CONST019 * VAR06 + CONST019 * VAR24 + VAR17 * (CONST013 * VAR08 -
        CONST013 * VAR26)
    Y07 = y * (CONST011 * VAR08 * z - CONST017 * VAR25)
    Y08 = CONST005 * VAR06 + CONST005 * VAR24 + CONST012 * VAR08 * VAR26
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = (output_striding + block_size * output_stride *
        block_id + col_offset)
    tl.store(output_ptr + output_row_offset, Y00, mask=output_row_offset <
        output_numel)
    tl.store(output_ptr + output_row_offset + 1, Y01, mask=
        output_row_offset + 1 < output_numel)
    tl.store(output_ptr + output_row_offset + 2, Y02, mask=
        output_row_offset + 2 < output_numel)
    tl.store(output_ptr + output_row_offset + 3, Y03, mask=
        output_row_offset + 3 < output_numel)
    tl.store(output_ptr + output_row_offset + 4, Y04, mask=
        output_row_offset + 4 < output_numel)
    tl.store(output_ptr + output_row_offset + 5, Y05, mask=
        output_row_offset + 5 < output_numel)
    tl.store(output_ptr + output_row_offset + 6, Y06, mask=
        output_row_offset + 6 < output_numel)
    tl.store(output_ptr + output_row_offset + 7, Y07, mask=
        output_row_offset + 7 < output_numel)
    tl.store(output_ptr + output_row_offset + 8, Y08, mask=
        output_row_offset + 8 < output_numel)
