import triton
import triton.language as tl
import torch

@triton.jit
def eighth_order_fwd(coord_ptr: 'tl.tensor', output_ptr: 'tl.tensor',
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
    CONST000 = 1.12741169450483
    CONST003 = 4.12310562561766
    CONST004 = 4.50964677801932
    CONST006 = 6.76447016702898
    CONST007 = 1.69594242329302
    CONST008 = 1.88707052233084
    CONST010 = 2.58397773170915
    CONST011 = 13.136713523081
    CONST012 = 13.136713523081
    CONST014 = -489.184589393411
    CONST015 = 24.738633753706
    CONST017 = 24.738633753706
    CONST019 = 48.9184589393411
    CONST020 = 48.5105296237322
    CONST021 = 51.744564931981
    CONST024 = 65.6835676154051
    CONST025 = 67.8376969317208
    CONST029 = 97.0210592474644
    CONST030 = -6.78376969317208
    CONST031 = 103.489129863962
    CONST032 = -407.026181590325
    CONST033 = 108.231522672464
    CONST035 = 110.066532613517
    CONST036 = 110.066532613517
    CONST037 = -396.284809689477
    CONST040 = -361.756882439281
    CONST041 = -1.88707052233084
    CONST042 = 158.513923875791
    CONST045 = 180.87844121964
    CONST046 = 194.042118494929
    CONST047 = -12.2296147348353
    CONST048 = 203.513090795162
    CONST050 = 216.463045344927
    CONST051 = 217.054129463568
    CONST052 = 216.463045344927
    CONST053 = -6.78376969317208
    CONST054 = -271.350787726883
    CONST055 = 244.592294696706
    CONST056 = 244.592294696706
    CONST057 = -262.734270461621
    CONST058 = -258.722824659905
    CONST061 = -217.054129463568
    CONST062 = -210.187416369296
    CONST063 = -175.156180307747
    CONST064 = -162.81047263613
    CONST066 = -144.702752975712
    CONST067 = -129.877827206956
    CONST068 = -129.361412329953
    CONST070 = -108.231522672464
    CONST071 = -108.231522672464
    CONST072 = -87.5780901538735
    CONST073 = -3.23403530824881
    CONST074 = -72.3513764878561
    CONST075 = -70.0624721230988
    CONST076 = -65.6835676154052
    CONST077 = -61.1480736741764
    CONST078 = -61.1480736741764
    CONST079 = -57.7234787586472
    CONST080 = -57.7234787586472
    CONST081 = -51.744564931981
    CONST082 = -48.5105296237322
    CONST083 = -40.5868210021738
    CONST084 = -39.4101405692431
    CONST085 = -40.7026181590325
    CONST086 = -36.0771742241545
    CONST087 = -36.0771742241545
    CONST088 = -26.4189873126318
    CONST089 = -20.6718218536732
    CONST090 = -528.379746252636
    CONST091 = -16.9594242329302
    CONST092 = -13.136713523081
    CONST093 = -12.2296147348353
    CONST094 = -11.3224231339851
    CONST095 = -10.3359109268366
    CONST096 = -9.70210592474644
    CONST097 = -11.3224231339851
    CONST098 = -13.5289403340579
    CONST099 = -6.78376969317208
    CONST100 = -13.5289403340579
    CONST101 = -13.136713523081
    CONST102 = -3.23403530824881
    CONST103 = -1.61701765412441
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR02 = VAR06 * VAR06
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR11 = VAR15 * VAR16
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR14 = VAR16 * VAR17
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR20 = VAR24 * VAR24
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    VAR23 = VAR25 * VAR26
    Y00 = (-CONST066 * VAR05 * VAR25 + CONST066 * VAR07 * VAR23 + CONST089 *
        VAR03 * z - CONST089 * VAR21 * x)
    Y01 = y * (CONST040 * VAR07 * VAR24 + CONST051 * VAR05 * VAR26 - 
        CONST074 * VAR22 * x + CONST095 * VAR03)
    Y02 = CONST097 * VAR03 * z + VAR05 * (CONST042 * VAR17 * z - CONST088 *
        VAR25) + VAR07 * (-CONST088 * VAR23 + CONST090 * VAR17 * VAR25) + x * (
        CONST042 * VAR17 * VAR23 + CONST094 * VAR21)
    Y03 = VAR16 * (CONST014 * VAR07 * VAR26 + CONST019 * VAR05 + CONST055 *
        VAR24 * x) + y * (CONST035 * VAR05 * VAR26 + CONST077 * VAR22 * x -
        CONST078 * VAR07 * VAR24 + CONST093 * VAR03)
    Y04 = CONST099 * VAR03 * z + VAR05 * (-CONST064 * VAR17 * z + CONST099 *
        VAR25) + VAR07 * (-CONST053 * VAR23 + CONST054 * VAR15 * z) + x * (
        -CONST053 * VAR21 - CONST054 * VAR15 * VAR25 + CONST064 * VAR17 * VAR23
        )
    Y05 = VAR14 * (-CONST062 * VAR26 * x + CONST075 * VAR07) + VAR16 * (
        CONST057 * VAR24 * x + CONST063 * VAR07 * VAR26 - CONST072 * VAR05
        ) + y * (CONST011 * VAR05 * VAR26 + CONST024 * VAR07 * VAR24 - 
        CONST084 * VAR22 * x + CONST092 * VAR03)
    Y06 = CONST102 * VAR03 * z + VAR05 * (CONST029 * VAR17 * z + CONST096 *
        VAR25) + VAR07 * (CONST046 * VAR17 * VAR25 + CONST058 * VAR15 * z +
        CONST096 * VAR23) + x * (CONST029 * VAR17 * VAR23 + CONST031 *
        VAR13 * z + CONST058 * VAR15 * VAR25 + CONST102 * VAR21)
    Y07 = CONST098 * VAR03 * y + VAR05 * (CONST033 * VAR16 + CONST083 *
        VAR26 * y) + VAR07 * (CONST050 * VAR16 * VAR26 + CONST067 * VAR14 +
        CONST083 * VAR24 * y) + x * (CONST015 * VAR12 + CONST067 * VAR14 *
        VAR26 - CONST070 * VAR16 * VAR24 + CONST098 * VAR22 * y)
    Y08 = (CONST000 * VAR02 + CONST000 * VAR20 + CONST003 * VAR11 - 
        CONST070 * VAR15 * VAR24 + CONST080 * VAR13 * VAR26 + CONST087 *
        VAR17 * VAR22 + VAR04 * (CONST004 * VAR26 + CONST086 * VAR17) + 
        VAR06 * (CONST006 * VAR24 - CONST070 * VAR15 + CONST071 * VAR17 *
        VAR26) + VAR08 * (CONST004 * VAR22 + CONST050 * VAR15 * VAR26 + 
        CONST070 * VAR17 * VAR24 + CONST079 * VAR13))
    Y09 = CONST098 * VAR21 * y + VAR23 * (CONST033 * VAR16 + CONST083 *
        VAR08 * y) + VAR25 * (CONST052 * VAR08 * VAR16 + CONST067 * VAR14 +
        CONST083 * VAR06 * y) + z * (CONST017 * VAR12 + CONST033 * VAR06 *
        VAR16 + CONST067 * VAR08 * VAR14 + CONST100 * VAR04 * y)
    Y10 = (CONST073 * VAR08 * VAR22 - CONST102 * VAR04 * VAR26 - CONST103 *
        VAR02 + CONST103 * VAR20 + VAR13 * (CONST021 * VAR26 + CONST081 *
        VAR08) + VAR15 * (-CONST068 * VAR06 + CONST068 * VAR24) + VAR17 * (
        CONST020 * VAR08 * VAR24 + CONST020 * VAR22 + CONST082 * VAR04 + 
        CONST082 * VAR06 * VAR26))
    Y11 = VAR14 * (CONST062 * VAR08 * z - CONST075 * VAR25) + VAR16 * (-
        CONST057 * VAR06 * z - CONST063 * VAR08 * VAR25 + CONST072 * VAR23
        ) + y * (CONST012 * VAR21 + CONST076 * VAR06 * VAR25 + CONST084 *
        VAR04 * z + CONST101 * VAR08 * VAR23)
    Y12 = (CONST007 * VAR02 + CONST007 * VAR20 + CONST030 * VAR04 * VAR26 +
        CONST053 * VAR08 * VAR22 + CONST091 * VAR06 * VAR24 + VAR15 * (
        CONST025 * VAR06 + CONST025 * VAR24 + CONST032 * VAR08 * VAR26) + 
        VAR17 * (CONST048 * VAR06 * VAR26 + CONST048 * VAR08 * VAR24 + 
        CONST085 * VAR04 + CONST085 * VAR22))
    Y13 = VAR16 * (CONST014 * VAR08 * VAR25 + CONST019 * VAR23 + CONST056 *
        VAR06 * z) + y * (CONST036 * VAR08 * VAR23 + CONST047 * VAR21 - 
        CONST077 * VAR06 * VAR25 + CONST078 * VAR04 * z)
    Y14 = (CONST008 * VAR02 + CONST041 * VAR20 + CONST088 * VAR04 * VAR26 -
        CONST088 * VAR08 * VAR22 + VAR17 * (-CONST037 * VAR06 * VAR26 + 
        CONST037 * VAR08 * VAR24 + CONST088 * VAR04 - CONST088 * VAR22))
    Y15 = y * (-CONST040 * VAR06 * VAR25 + CONST061 * VAR08 * VAR23 + 
        CONST074 * VAR04 * z - CONST095 * VAR21)
    Y16 = (CONST010 * VAR02 + CONST010 * VAR20 + CONST045 * VAR06 * VAR24 +
        CONST074 * VAR04 * VAR26 + CONST074 * VAR08 * VAR22)
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
    tl.store(output_ptr + output_row_offset + 15, Y15, mask=
        output_row_offset + 15 < output_numel)
    tl.store(output_ptr + output_row_offset + 16, Y16, mask=
        output_row_offset + 16 < output_numel)
