import triton
import triton.language as tl
import torch

@triton.jit
def seventh_order_fwd(coord_ptr: 'tl.tensor', output_ptr: 'tl.tensor',
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
    CONST002 = 3.87298334620742
    CONST008 = 11.7655316231354
    CONST010 = 16.5555704843566
    CONST012 = 20.4939015319192
    CONST013 = 20.4939015319192
    CONST014 = 22.0740939791422
    CONST015 = 23.5310632462709
    CONST017 = 36.7901566319036
    CONST019 = 38.4260653723485
    CONST020 = 38.4260653723485
    CONST021 = 38.4260653723485
    CONST023 = -4.9916923169903
    CONST025 = 47.0621264925418
    CONST026 = 50.8329064189723
    CONST028 = 55.1852349478554
    CONST029 = 56.2781179722634
    CONST030 = 56.2781179722634
    CONST032 = 66.5558975598707
    CONST033 = 75.2994023880668
    CONST037 = 101.665812837945
    CONST038 = 110.370469895711
    CONST041 = 147.160626527614
    CONST042 = -1.66389743899677
    CONST043 = -9.37968632871057
    CONST044 = -1.66389743899677
    CONST045 = -220.740939791422
    CONST046 = -220.740939791422
    CONST047 = -1.60108605718119
    CONST048 = -187.593726574211
    CONST049 = -9.1975391579759
    CONST050 = -1.83950783159518
    CONST051 = -1.83950783159518
    CONST052 = -4.80325817154356
    CONST053 = -147.160626527614
    CONST054 = -140.695294930659
    CONST055 = -133.111795119741
    CONST056 = -125.499003980111
    CONST057 = -125.499003980111
    CONST058 = -99.833846339806
    CONST059 = -87.7389315936062
    CONST060 = -76.852130744697
    CONST061 = -66.5558975598707
    CONST062 = -62.7495019900557
    CONST063 = -52.6433589561637
    CONST064 = -44.1481879582843
    CONST065 = -44.3705983732471
    CONST066 = -40.6663251351779
    CONST067 = -40.6663251351779
    CONST068 = -8.31948719498384
    CONST069 = -37.6497011940334
    CONST070 = -33.2779487799353
    CONST071 = -25.4164532094862
    CONST072 = -25.4164532094862
    CONST073 = -17.5477863187212
    CONST074 = -11.7655316231354
    CONST075 = -11.0370469895711
    CONST076 = -9.1975391579759
    CONST077 = -8.47215106982872
    CONST078 = -4.80325817154356
    CONST079 = -2.50682661696018
    CONST080 = -1.60108605718119
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR14 = VAR16 * VAR17
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    VAR23 = VAR25 * VAR26
    Y00 = (CONST059 * VAR07 * VAR24 - CONST063 * VAR05 * VAR26 - CONST073 *
        VAR22 * x + CONST079 * VAR03)
    Y01 = y * (CONST029 * VAR23 * x + CONST030 * VAR05 * z + CONST048 *
        VAR07 * VAR25)
    Y02 = CONST050 * VAR03 + VAR05 * (CONST010 * VAR26 + CONST014 * VAR17
        ) + VAR07 * (CONST045 * VAR17 * VAR26 - CONST076 * VAR24) + x * (
        CONST038 * VAR17 * VAR24 + CONST076 * VAR22)
    Y03 = VAR16 * (CONST041 * VAR25 * x + CONST053 * VAR07 * z) + y * (-
        CONST064 * VAR05 * z + CONST064 * VAR23 * x)
    Y04 = CONST042 * VAR03 + VAR05 * (-CONST042 * VAR26 - CONST070 * VAR17
        ) + VAR07 * (CONST061 * VAR17 * VAR26 + CONST065 * VAR15 - CONST068 *
        VAR24) + x * (-CONST023 * VAR22 - CONST055 * VAR15 * VAR26 + 
        CONST058 * VAR17 * VAR24)
    Y05 = CONST015 * VAR05 * y * z + VAR07 * (CONST025 * VAR25 * y + 
        CONST057 * VAR16 * z) + x * (CONST015 * VAR23 * y + CONST033 *
        VAR14 * z + CONST056 * VAR16 * VAR25)
    Y06 = CONST047 * VAR03 + VAR05 * (CONST020 * VAR17 + CONST078 * VAR26
        ) + VAR07 * (CONST052 * VAR24 + CONST060 * VAR15 - CONST060 * VAR17 *
        VAR26) + x * (CONST012 * VAR13 + CONST019 * VAR17 * VAR24 + 
        CONST060 * VAR15 * VAR26 + CONST080 * VAR22)
    Y07 = CONST002 * VAR12 + VAR14 * (CONST066 * VAR08 + CONST067 * VAR26
        ) + VAR16 * (CONST026 * VAR06 + CONST026 * VAR24 + CONST037 * VAR08 *
        VAR26) + y * (CONST071 * VAR06 * VAR26 + CONST072 * VAR08 * VAR24 +
        CONST077 * VAR04 + CONST077 * VAR22)
    Y08 = CONST047 * VAR21 + VAR23 * (CONST020 * VAR17 + CONST052 * VAR08
        ) + VAR25 * (CONST052 * VAR06 - CONST060 * VAR08 * VAR17 + CONST060 *
        VAR15) + z * (CONST013 * VAR13 + CONST021 * VAR06 * VAR17 + 
        CONST047 * VAR04 + CONST060 * VAR08 * VAR15)
    Y09 = VAR14 * (CONST069 * VAR08 - CONST069 * VAR26) + VAR16 * (-
        CONST062 * VAR06 + CONST062 * VAR24) + y * (CONST008 * VAR08 *
        VAR24 + CONST074 * VAR04 + CONST074 * VAR06 * VAR26 - CONST074 * VAR22)
    Y10 = -CONST042 * VAR21 + VAR23 * (CONST044 * VAR08 + CONST070 * VAR17
        ) + VAR25 * (CONST032 * VAR08 * VAR17 - CONST065 * VAR15 + CONST068 *
        VAR06) + z * (CONST023 * VAR04 + CONST055 * VAR08 * VAR15 - 
        CONST058 * VAR06 * VAR17)
    Y11 = VAR16 * (CONST017 * VAR06 + CONST017 * VAR24 + CONST046 * VAR08 *
        VAR26) + y * (CONST028 * VAR06 * VAR26 + CONST028 * VAR08 * VAR24 +
        CONST075 * VAR04 + CONST075 * VAR22)
    Y12 = CONST051 * VAR21 + VAR23 * (CONST010 * VAR08 + CONST014 * VAR17
        ) + VAR25 * (CONST045 * VAR08 * VAR17 - CONST049 * VAR06) + z * (
        CONST038 * VAR06 * VAR17 + CONST049 * VAR04)
    Y13 = y * (CONST043 * VAR04 - CONST043 * VAR22 - CONST054 * VAR06 *
        VAR26 + CONST054 * VAR08 * VAR24)
    Y14 = (-CONST059 * VAR06 * VAR25 + CONST063 * VAR08 * VAR23 + CONST073 *
        VAR04 * z - CONST079 * VAR21)
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
    tl.store(output_ptr + output_row_offset + 13, Y13, mask=
        output_row_offset + 13 < output_numel)
    tl.store(output_ptr + output_row_offset + 14, Y14, mask=
        output_row_offset + 14 < output_numel)
