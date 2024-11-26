import triton
import triton.language as tl
import torch

@triton.jit
def third_order_fwd(coord_ptr: 'tl.tensor', output_ptr: 'tl.tensor',
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
    CONST000 = 2.64575131106459
    CONST002 = 5.1234753829798
    CONST004 = 6.48074069840786
    CONST005 = 10.2469507659596
    CONST006 = -2.09165006633519
    CONST007 = -1
    CONST008 = -6.27495019900557
    CONST009 = -3.96862696659689
    CONST010 = -1.62018517460197
    VAR07 = x * x * x
    VAR08 = x * x
    VAR16 = y * y * y
    VAR17 = y * y
    VAR25 = z * z * z
    VAR26 = z * z
    Y00 = CONST006 * VAR07 - CONST008 * VAR26 * x
    Y01 = CONST005 * x * y * z
    Y02 = CONST010 * VAR07 + x * (CONST004 * VAR17 + CONST010 * VAR26)
    Y03 = CONST000 * VAR16 + CONST009 * VAR08 * y + CONST009 * VAR26 * y
    Y04 = CONST010 * VAR25 + z * (CONST004 * VAR17 + CONST010 * VAR08)
    Y05 = CONST002 * y * (CONST007 * VAR08 + VAR26)
    Y06 = -CONST006 * VAR25 + CONST008 * VAR08 * z
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
