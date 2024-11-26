import triton
import triton.language as tl
import torch

@triton.jit
def fifth_order_fwd(coord_ptr: 'tl.tensor', output_ptr: 'tl.tensor',
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
    CONST000 = 1.73430461568895
    CONST001 = 2.32681380862329
    CONST002 = 1.60565407233314
    CONST003 = 3.21130814466628
    CONST004 = 3.3166247903554
    CONST005 = 6.21867148191637
    CONST006 = 6.21867148191637
    CONST007 = 1.60565407233314
    CONST009 = 11.6340690431164
    CONST010 = 12.8452325786651
    CONST011 = 12.4373429638327
    CONST012 = 12.8452325786651
    CONST013 = 13.8744369255116
    CONST017 = 33.9852909359329
    CONST018 = 7.35803132638072
    CONST020 = -44.1481879582843
    CONST021 = -41.6233107765348
    CONST022 = -29.4321253055229
    CONST023 = -23.2681380862329
    CONST024 = -19.2678488679977
    CONST025 = -19.2678488679977
    CONST026 = -16.9926454679664
    CONST027 = -16.9926454679664
    CONST028 = -13.8744369255116
    CONST029 = -16.583123951777
    CONST030 = 3.4686092313779
    CONST031 = -8.49632273398321
    CONST032 = -5.20291384706685
    CONST033 = -3.4686092313779
    CONST034 = -1.73430461568895
    VAR05 = x * x * x * x * x
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR14 = y * y * y * y * y
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR23 = z * z * z * z * z
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    Y00 = CONST001 * VAR05 + CONST009 * VAR24 * x + CONST023 * VAR07 * VAR26
    Y01 = y * (CONST022 * VAR07 * z - CONST022 * VAR25 * x)
    Y02 = CONST000 * VAR05 + VAR07 * (CONST028 * VAR17 + CONST033 * VAR26
        ) + x * (-CONST021 * VAR17 * VAR26 + CONST032 * VAR24)
    Y03 = CONST027 * VAR07 * y * z + x * (CONST017 * VAR16 * z + CONST026 *
        VAR25 * y)
    Y04 = CONST002 * VAR05 + VAR07 * (CONST003 * VAR26 + CONST025 * VAR17
        ) + x * (CONST002 * VAR24 + CONST010 * VAR15 + CONST024 * VAR17 * VAR26
        )
    Y05 = CONST004 * VAR14 + VAR16 * (CONST029 * VAR08 + CONST029 * VAR26
        ) + y * (CONST005 * VAR06 + CONST006 * VAR24 + CONST011 * VAR08 * VAR26
        )
    Y06 = CONST002 * VAR23 + VAR25 * (CONST003 * VAR08 + CONST024 * VAR17
        ) + z * (CONST007 * VAR06 + CONST012 * VAR15 + CONST024 * VAR08 * VAR17
        )
    Y07 = VAR16 * (CONST026 * VAR08 - CONST026 * VAR26) + y * (-CONST031 *
        VAR06 + CONST031 * VAR24)
    Y08 = CONST034 * VAR23 + VAR25 * (CONST013 * VAR17 + CONST030 * VAR08
        ) + z * (CONST021 * VAR08 * VAR17 - CONST032 * VAR06)
    Y09 = y * (CONST018 * VAR06 + CONST018 * VAR24 + CONST020 * VAR08 * VAR26)
    Y10 = CONST001 * VAR23 + CONST009 * VAR06 * z + CONST023 * VAR08 * VAR25
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
    tl.store(output_ptr + output_row_offset + 9, Y09, mask=
        output_row_offset + 9 < output_numel)
    tl.store(output_ptr + output_row_offset + 10, Y10, mask=
        output_row_offset + 10 < output_numel)
