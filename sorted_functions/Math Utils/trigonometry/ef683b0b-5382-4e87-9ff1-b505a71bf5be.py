import triton
import triton.language as tl
import torch

@triton.jit
def sixth_order_fwd(coord_ptr: 'tl.tensor', output_ptr: 'tl.tensor',
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
    CONST002 = 3.26558761940328
    CONST003 = 3.26558761940328
    CONST004 = 6.53117523880657
    CONST006 = 8.38944649544891
    CONST007 = 9.79676285820985
    CONST008 = 10.3266947761614
    CONST009 = 3.60555127546399
    CONST010 = -1.78863600265677
    CONST011 = 14.5309475774982
    CONST012 = 8.94318001328386
    CONST013 = 16.5227116418583
    CONST014 = 16.5227116418583
    CONST015 = 17.8863600265677
    CONST017 = 20.6533895523229
    CONST018 = 20.2812259244849
    CONST019 = -107.318160159406
    CONST020 = 17.8863600265677
    CONST022 = 29.3902885746295
    CONST024 = 40.5624518489699
    CONST025 = 41.9472324772445
    CONST026 = -1.63279380970164
    CONST027 = -83.8944649544891
    CONST028 = -78.3741028656788
    CONST030 = -71.5454401062709
    CONST032 = -52.2494019104525
    CONST033 = -52.2494019104525
    CONST035 = -48.4364919249939
    CONST036 = -41.3067791046458
    CONST037 = -36.3273689437454
    CONST038 = -29.3902885746295
    CONST039 = -27.0416345659799
    CONST040 = -26.1247009552263
    CONST041 = -26.1247009552263
    CONST042 = -19.5935257164197
    CONST043 = -2.4218245962497
    CONST044 = -9.79676285820985
    CONST045 = -7.15454401062709
    CONST046 = -3.38020432074749
    CONST047 = -1.1267347735825
    VAR07 = x * x * x
    VAR08 = x * x
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR06 = VAR08 * VAR08
    VAR16 = y * y * y
    VAR17 = y * y
    VAR13 = VAR16 * VAR16
    VAR14 = VAR16 * VAR17
    VAR15 = VAR17 * VAR17
    VAR25 = z * z * z
    VAR26 = z * z
    VAR22 = VAR25 * VAR25
    VAR23 = VAR25 * VAR26
    VAR24 = VAR26 * VAR26
    Y00 = (CONST011 * VAR05 * z + CONST011 * VAR23 * x + CONST035 * VAR07 *
        VAR25)
    Y01 = y * (CONST006 * VAR05 + CONST025 * VAR24 * x + CONST027 * VAR07 *
        VAR26)
    Y02 = -CONST045 * VAR05 * z + CONST045 * VAR23 * x + VAR17 * (CONST030 *
        VAR07 * z - CONST030 * VAR25 * x)
    Y03 = VAR16 * (-CONST028 * VAR26 * x + CONST040 * VAR07) + y * (
        CONST007 * VAR05 + CONST038 * VAR24 * x + CONST042 * VAR07 * VAR26)
    Y04 = CONST003 * VAR05 * z + VAR07 * (CONST004 * VAR25 + CONST033 *
        VAR17 * z) + x * (CONST002 * VAR23 - CONST032 * VAR15 * z + 
        CONST032 * VAR17 * VAR25)
    Y05 = CONST008 * VAR05 * y + VAR07 * (CONST017 * VAR26 * y + CONST036 *
        VAR16) + x * (CONST008 * VAR24 * y + CONST013 * VAR14 + CONST036 *
        VAR16 * VAR26)
    Y06 = (CONST009 * VAR13 + CONST018 * VAR17 * VAR24 + CONST039 * VAR15 *
        VAR26 + CONST047 * VAR04 + CONST047 * VAR22 + VAR06 * (CONST018 *
        VAR17 + CONST046 * VAR26) + VAR08 * (CONST024 * VAR17 * VAR26 + 
        CONST039 * VAR15 + CONST046 * VAR24))
    Y07 = CONST008 * VAR23 * y + VAR25 * (CONST017 * VAR08 * y + CONST036 *
        VAR16) + z * (CONST008 * VAR06 * y + CONST014 * VAR14 + CONST036 *
        VAR08 * VAR16)
    Y08 = (CONST026 * VAR04 - CONST026 * VAR22 + CONST040 * VAR17 * VAR24 -
        CONST041 * VAR15 * VAR26 + VAR06 * (CONST026 * VAR26 - CONST041 *
        VAR17) + VAR08 * (-CONST026 * VAR24 + CONST041 * VAR15))
    Y09 = VAR16 * (CONST028 * VAR08 * z - CONST041 * VAR25) + y * (CONST022 *
        VAR06 * z - CONST042 * VAR08 * VAR25 + CONST044 * VAR23)
    Y10 = (CONST010 * VAR04 + CONST010 * VAR22 + CONST020 * VAR17 * VAR24 +
        VAR06 * (CONST012 * VAR26 + CONST015 * VAR17) + VAR08 * (CONST012 *
        VAR24 + CONST019 * VAR17 * VAR26))
    Y11 = y * (CONST006 * VAR23 + CONST025 * VAR06 * z + CONST027 * VAR08 *
        VAR25)
    Y12 = (-CONST037 * VAR06 * VAR26 + CONST037 * VAR08 * VAR24 + CONST043 *
        VAR04 - CONST043 * VAR22)
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
    tl.store(output_ptr + output_row_offset + 11, Y11, mask=
        output_row_offset + 11 < output_numel)
    tl.store(output_ptr + output_row_offset + 12, Y12, mask=
        output_row_offset + 12 < output_numel)
