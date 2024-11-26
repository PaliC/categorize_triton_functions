import triton
import triton.language as tl
import torch

@triton.jit
def sixth_order_bwd(coord_ptr: 'tl.tensor', coord_grad_ptr: 'tl.tensor',
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
    g_7 = tl.load(sph_grad_ptr + output_row_offset + 7, mask=
        output_row_offset + 7 < output_numel)
    g_8 = tl.load(sph_grad_ptr + output_row_offset + 8, mask=
        output_row_offset + 8 < output_numel)
    g_9 = tl.load(sph_grad_ptr + output_row_offset + 9, mask=
        output_row_offset + 9 < output_numel)
    g_10 = tl.load(sph_grad_ptr + output_row_offset + 10, mask=
        output_row_offset + 10 < output_numel)
    g_11 = tl.load(sph_grad_ptr + output_row_offset + 11, mask=
        output_row_offset + 11 < output_numel)
    g_12 = tl.load(sph_grad_ptr + output_row_offset + 12, mask=
        output_row_offset + 12 < output_numel)
    CONST000 = 2.0
    CONST002 = 4.0
    CONST003 = 3.0
    CONST004 = 6.53117523880657
    CONST006 = 8.94318001328386
    CONST007 = 8.38944649544891
    CONST008 = 10.3266947761614
    CONST009 = 9.79676285820985
    CONST013 = 16.3279380970164
    CONST014 = 17.8863600265677
    CONST015 = 16.5227116418583
    CONST016 = 20.6533895523229
    CONST017 = 20.2812259244849
    CONST018 = 21.6333076527839
    CONST020 = 17.8863600265677
    CONST022 = 29.3902885746295
    CONST024 = 35.7727200531355
    CONST026 = 40.5624518489699
    CONST028 = 41.9472324772445
    CONST029 = 48.9838142910493
    CONST030 = 51.6334738808072
    CONST035 = 71.5454401062709
    CONST037 = 81.1249036979398
    CONST039 = 82.6135582092915
    CONST040 = -3.26558761940328
    CONST042 = 117.561154298518
    CONST046 = 208.99760764181
    CONST048 = -251.683394863467
    CONST049 = -214.636320318813
    CONST050 = -214.636320318813
    CONST051 = 16.5227116418583
    CONST052 = -167.788929908978
    CONST053 = -156.748205731358
    CONST054 = -145.309475774982
    CONST055 = -123.920337313937
    CONST056 = -117.561154298518
    CONST057 = 3.26558761940328
    CONST058 = -108.16653826392
    CONST059 = -107.318160159406
    CONST060 = -104.498803820905
    CONST061 = -104.498803820905
    CONST062 = -83.8944649544891
    CONST063 = -82.6135582092915
    CONST064 = -78.3741028656788
    CONST065 = -72.6547378874909
    CONST066 = -71.5454401062709
    CONST067 = -58.7805771492591
    CONST068 = -54.0832691319598
    CONST069 = -52.2494019104525
    CONST070 = -52.2494019104525
    CONST071 = -48.9838142910492
    CONST072 = -41.3067791046458
    CONST073 = -39.1870514328394
    CONST074 = -35.7727200531355
    CONST075 = -29.3902885746295
    CONST076 = -27.0416345659799
    CONST077 = -26.1247009552263
    CONST078 = -26.1247009552263
    CONST079 = -19.5935257164197
    CONST080 = -14.5309475774982
    CONST081 = -13.52081728299
    CONST082 = -10.7318160159406
    CONST083 = -9.79676285820985
    CONST084 = -7.15454401062709
    CONST085 = -6.76040864149498
    CONST086 = -3.38020432074749
    CONST087 = -1.63279380970164
    VAR07 = x * x * x
    VAR08 = x * x
    VAR05 = VAR07 * VAR08
    VAR06 = VAR08 * VAR08
    VAR16 = y * y * y
    VAR17 = y * y
    VAR14 = VAR16 * VAR17
    VAR15 = VAR17 * VAR17
    VAR25 = z * z * z
    VAR26 = z * z
    VAR23 = VAR25 * VAR26
    VAR24 = VAR26 * VAR26
    g_x = tl.load(coord_grad_ptr + coord_row_offset, mask=coord_row_offset <
        coord_numel)
    g_y = tl.load(coord_grad_ptr + coord_row_offset + 1, mask=
        coord_row_offset + 1 < coord_numel)
    g_z = tl.load(coord_grad_ptr + coord_row_offset + 2, mask=
        coord_row_offset + 2 < coord_numel)
    g_x += g_0 * (CONST054 * VAR08 * VAR25 - CONST065 * VAR06 * z - 
        CONST080 * VAR23) + g_1 * y * (CONST028 * VAR06 + CONST028 * VAR24 +
        CONST048 * VAR08 * VAR26) + g_10 * (CONST000 * x * (CONST006 *
        VAR24 + CONST059 * VAR17 * VAR26) + CONST002 * VAR07 * (CONST006 *
        VAR26 + CONST014 * VAR17) + CONST082 * VAR05) + g_11 * y * (-
        CONST052 * VAR07 * z + CONST052 * VAR25 * x) + g_12 * (-CONST054 *
        VAR07 * VAR26 + CONST065 * VAR24 * x + CONST080 * VAR05) + g_2 * (-
        CONST074 * VAR06 * z + CONST084 * VAR23 + VAR17 * (CONST049 * VAR08 *
        z - CONST066 * VAR25)) + g_3 * (VAR16 * (CONST064 * VAR08 - 
        CONST064 * VAR26) + y * (CONST029 * VAR06 + CONST067 * VAR08 *
        VAR26 + CONST075 * VAR24)) + g_4 * (CONST003 * VAR08 * (CONST004 *
        VAR25 + CONST069 * VAR17 * z) + CONST013 * VAR06 * z - CONST040 *
        VAR23 - CONST070 * VAR15 * z + CONST070 * VAR17 * VAR25) + g_5 * (
        CONST003 * VAR08 * (CONST016 * VAR26 * y + CONST072 * VAR16) + 
        CONST008 * VAR24 * y + CONST015 * VAR14 + CONST030 * VAR06 * y + 
        CONST072 * VAR16 * VAR26) + g_6 * (CONST000 * x * (CONST026 * VAR17 *
        VAR26 + CONST076 * VAR15 + CONST086 * VAR24) + CONST002 * VAR07 * (
        CONST017 * VAR17 + CONST086 * VAR26) + CONST085 * VAR05) + g_7 * (-
        CONST072 * VAR25 * x * y + z * (CONST063 * VAR16 * x - CONST072 *
        VAR07 * y)) + g_8 * (CONST000 * x * (CONST077 * VAR15 - CONST087 *
        VAR24) + CONST002 * VAR07 * (-CONST077 * VAR17 + CONST087 * VAR26) +
        CONST083 * VAR05) + g_9 * (CONST053 * VAR16 * x * z + y * (CONST042 *
        VAR07 * z - CONST073 * VAR25 * x))
    g_y += CONST000 * g_2 * y * (CONST066 * VAR07 * z - CONST066 * VAR25 * x
        ) + g_1 * (CONST007 * VAR05 + CONST028 * VAR24 * x + CONST062 *
        VAR07 * VAR26) + g_10 * (CONST024 * VAR06 * y + CONST050 * VAR08 *
        VAR26 * y - CONST074 * VAR24 * y) + g_11 * (CONST007 * VAR23 + 
        CONST028 * VAR06 * z + CONST062 * VAR08 * VAR25) + g_3 * (CONST003 *
        VAR17 * (-CONST064 * VAR26 * x + CONST078 * VAR07) + CONST009 *
        VAR05 + CONST075 * VAR24 * x + CONST079 * VAR07 * VAR26) + g_4 * (
        CONST061 * VAR07 * y * z + x * (CONST046 * VAR16 * z + CONST060 *
        VAR25 * y)) + g_5 * (CONST008 * VAR05 + VAR07 * (CONST016 * VAR26 +
        CONST055 * VAR17) + x * (CONST008 * VAR24 + CONST055 * VAR17 *
        VAR26 - CONST063 * VAR15)) + g_6 * (CONST018 * VAR14 + CONST026 *
        VAR06 * y + CONST026 * VAR24 * y + CONST058 * VAR16 * VAR26 + VAR08 *
        (CONST037 * VAR26 * y + CONST058 * VAR16)) + g_7 * (CONST008 *
        VAR23 + VAR25 * (CONST016 * VAR08 + CONST055 * VAR17) + z * (
        CONST008 * VAR06 + CONST039 * VAR15 + CONST055 * VAR08 * VAR17)
        ) + g_8 * (CONST060 * VAR08 * VAR16 - CONST060 * VAR16 * VAR26 + 
        CONST069 * VAR24 * y - CONST070 * VAR06 * y) + g_9 * (CONST003 *
        VAR17 * (CONST064 * VAR08 * z - CONST077 * VAR25) + CONST022 *
        VAR06 * z - CONST079 * VAR08 * VAR25 + CONST083 * VAR23)
    g_z += g_0 * (CONST054 * VAR07 * VAR26 - CONST065 * VAR24 * x - 
        CONST080 * VAR05) + g_1 * y * (CONST052 * VAR07 * z - CONST052 *
        VAR25 * x) + g_10 * (CONST020 * VAR06 * z + CONST035 * VAR17 *
        VAR25 + CONST082 * VAR23 + VAR08 * (CONST050 * VAR17 * z - CONST074 *
        VAR25)) + g_11 * y * (CONST028 * VAR06 + CONST028 * VAR24 + 
        CONST048 * VAR08 * VAR26) + g_12 * (CONST054 * VAR08 * VAR25 - 
        CONST065 * VAR06 * z - CONST080 * VAR23) + g_2 * (CONST074 * VAR24 *
        x - CONST084 * VAR05 + VAR17 * (-CONST049 * VAR26 * x + CONST066 *
        VAR07)) + g_3 * (-CONST053 * VAR16 * x * z + y * (CONST056 * VAR25 *
        x + CONST073 * VAR07 * z)) + g_4 * (CONST057 * VAR05 + VAR07 * (
        CONST069 * VAR17 - CONST079 * VAR26) + x * (CONST013 * VAR24 + 
        CONST053 * VAR17 * VAR26 - CONST070 * VAR15)) + g_5 * (-CONST072 *
        VAR07 * y * z + x * (CONST063 * VAR16 * z - CONST072 * VAR25 * y)
        ) + g_6 * (CONST037 * VAR17 * VAR25 + CONST068 * VAR15 * z + 
        CONST085 * VAR06 * z + CONST085 * VAR23 + VAR08 * (CONST037 * VAR17 *
        z + CONST081 * VAR25)) + g_7 * (CONST003 * VAR26 * (CONST016 *
        VAR08 * y + CONST072 * VAR16) + CONST008 * VAR06 * y + CONST030 *
        VAR24 * y + CONST051 * VAR14 + CONST072 * VAR08 * VAR16) + g_8 * (
        CONST004 * VAR08 * VAR25 + CONST040 * VAR06 * z + CONST061 * VAR17 *
        VAR25 - CONST070 * VAR15 * z - CONST083 * VAR23) + g_9 * (VAR16 * (
        CONST064 * VAR08 - CONST064 * VAR26) + y * (CONST022 * VAR06 - 
        CONST067 * VAR08 * VAR26 + CONST071 * VAR24))
    tl.store(coord_grad_ptr + coord_row_offset, g_x, mask=coord_row_offset <
        coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 1, g_y, mask=
        coord_row_offset + 1 < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 2, g_z, mask=
        coord_row_offset + 2 < coord_numel)
