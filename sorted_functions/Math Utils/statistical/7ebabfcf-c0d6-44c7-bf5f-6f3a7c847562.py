import triton
import triton.language as tl
import torch

@triton.jit
def ninth_order_fwd(coord_ptr: 'tl.tensor', output_ptr: 'tl.tensor',
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
    CONST000 = 1.93163963757558
    CONST001 = 2.65478475211798
    CONST002 = 1.72771101506082
    CONST004 = 1.59908344719522
    CONST005 = 6.39633378878088
    CONST006 = 6.39633378878088
    CONST007 = 8.63855507530412
    CONST008 = 9.59450068317133
    CONST009 = 4.35889894354067
    CONST010 = 10.7269778688696
    CONST011 = 10.7269778688696
    CONST012 = 6.39633378878088
    CONST013 = 15.0007324039945
    CONST014 = 13.0937127087774
    CONST016 = 14.45506743704
    CONST017 = 14.45506743704
    CONST018 = 13.3827919767794
    CONST019 = 13.5214774630291
    CONST020 = 23.8930627690618
    CONST021 = 27.0429549260581
    CONST022 = 29.2403830344269
    CONST023 = 29.2403830344269
    CONST024 = 30.001464807989
    CONST025 = -480.023436927823
    CONST026 = -480.023436927823
    CONST029 = 42.9079114754785
    CONST030 = -462.562157985281
    CONST032 = -967.518168434061
    CONST034 = 57.8202697481601
    CONST035 = 58.9217071894985
    CONST036 = 58.9217071894985
    CONST037 = 62.4530292249704
    CONST038 = 1081.71819704233
    CONST039 = 64.3618672132178
    CONST040 = 578.202697481601
    CONST044 = 600.029296159779
    CONST045 = -936.795438374555
    CONST047 = 96.7518168434061
    CONST049 = 115.64053949632
    CONST051 = -392.811381263323
    CONST053 = 137.14955340795
    CONST055 = 150.007324039945
    CONST056 = -343.263291803828
    CONST058 = 11.2632978048796
    CONST061 = -315.37233853663
    CONST062 = -314.249105010659
    CONST063 = 205.957975082297
    CONST065 = -294.608535947493
    CONST066 = 240.011718463912
    CONST068 = 241.879542108515
    CONST069 = 255.853351551235
    CONST070 = 255.853351551235
    CONST071 = -241.879542108515
    CONST072 = -240.011718463912
    CONST073 = -241.879542108515
    CONST074 = 788.430846341574
    CONST075 = 1.72771101506082
    CONST076 = -1.93163963757558
    CONST077 = -1249.06058449941
    CONST078 = -223.00191917791
    CONST080 = -216.343639408465
    CONST081 = 300.01464807989
    CONST082 = -204.682681240988
    CONST083 = -204.682681240988
    CONST084 = -204.682681240988
    CONST086 = -196.405690631662
    CONST087 = -191.890013663426
    CONST088 = -191.890013663427
    CONST089 = -187.359087674911
    CONST090 = -693.843236977922
    CONST091 = 334.502878766866
    CONST092 = -176.765121568496
    CONST093 = -150.007324039945
    CONST094 = -144.5506743704
    CONST095 = 374.718175349822
    CONST096 = 374.718175349822
    CONST097 = -649.030918225395
    CONST099 = -630.744677073259
    CONST100 = -115.64053949632
    CONST101 = -114.421097267943
    CONST102 = -115.64053949632
    CONST103 = -104.74970167022
    CONST104 = 411.915950164594
    CONST105 = -95.5722510762473
    CONST106 = -90.106382439037
    CONST107 = -90.0043944239669
    CONST109 = -80.2967518606762
    CONST110 = -78.4601809837321
    CONST111 = 435.383175795327
    CONST112 = -589.217071894985
    CONST113 = -78.4601809837321
    CONST114 = 435.383175795328
    CONST115 = -68.5747767039748
    CONST116 = -63.9633378878088
    CONST117 = -63.9633378878088
    CONST118 = -62.4530292249704
    CONST119 = -58.9217071894985
    CONST120 = -1081.71819704233
    CONST121 = -57.8202697481601
    CONST122 = -57.8202697481601
    CONST123 = -58.9217071894985
    CONST124 = -54.0859098521163
    CONST125 = 462.562157985281
    CONST127 = -48.3759084217031
    CONST128 = -48.375908421703
    CONST129 = -38.6327927515116
    CONST130 = -30.9062342012093
    CONST131 = 483.759084217031
    CONST132 = -30.001464807989
    CONST133 = -30.001464807989
    CONST134 = -27.0429549260581
    CONST135 = -24.1879542108515
    CONST136 = -24.1879542108515
    CONST137 = -1.63671408859718
    CONST138 = -15.0007324039945
    CONST139 = -13.5214774630291
    CONST140 = -13.8216881204866
    CONST141 = -13.0937127087774
    CONST142 = -13.3827919767794
    CONST143 = -9.82028453158308
    CONST144 = -4.91014226579154
    CONST145 = 511.706703102471
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR01 = VAR07 * VAR07 * VAR07
    VAR02 = VAR06 * VAR06
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR10 = VAR16 * VAR16 * VAR16
    VAR11 = VAR15 * VAR15
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR14 = VAR16 * VAR17
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR19 = VAR25 * VAR25 * VAR25
    VAR20 = VAR24 * VAR24
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    VAR23 = VAR25 * VAR26
    Y00 = (CONST001 * VAR01 + CONST020 * VAR20 * x + CONST078 * VAR07 *
        VAR22 + CONST091 * VAR05 * VAR24 + CONST105 * VAR03 * VAR26)
    Y01 = y * (-CONST099 * VAR05 * VAR25 + CONST099 * VAR07 * VAR23 + 
        CONST106 * VAR03 * z - CONST106 * VAR21 * x)
    Y02 = CONST000 * VAR01 + VAR03 * (CONST129 * VAR26 + CONST130 * VAR17
        ) + VAR05 * (CONST021 * VAR24 - CONST097 * VAR17 * VAR26) + VAR07 * (
        CONST120 * VAR17 * VAR24 - CONST124 * VAR22) + x * (-CONST080 *
        VAR17 * VAR22 + CONST139 * VAR20)
    Y03 = VAR16 * (CONST077 * VAR07 * VAR25 + CONST095 * VAR05 * z + 
        CONST096 * VAR23 * x) + y * (-CONST089 * VAR05 * VAR25 - CONST089 *
        VAR07 * VAR23 + CONST109 * VAR03 * z + CONST109 * VAR21 * x)
    Y04 = (CONST002 * VAR01 + CONST007 * VAR20 * x + CONST135 * VAR05 *
        VAR24 + CONST140 * VAR03 * VAR26 + VAR15 * (CONST032 * VAR07 *
        VAR26 + CONST047 * VAR05 + CONST131 * VAR24 * x) + VAR17 * (-
        CONST071 * VAR07 * VAR24 + CONST071 * VAR22 * x + CONST111 * VAR05 *
        VAR26 + CONST127 * VAR03))
    Y05 = VAR14 * (CONST030 * VAR07 * z - CONST030 * VAR25 * x) + VAR16 * (
        CONST030 * VAR23 * x + CONST125 * VAR05 * z) + y * (CONST034 *
        VAR07 * VAR23 + CONST121 * VAR05 * VAR25 - CONST121 * VAR21 * x + 
        CONST122 * VAR03 * z)
    Y06 = CONST119 * VAR03 * VAR17 - CONST137 * VAR01 + VAR05 * (CONST035 *
        VAR17 * VAR26 - CONST086 * VAR15 + CONST143 * VAR24) + VAR07 * (
        CONST051 * VAR15 * VAR26 - CONST065 * VAR17 * VAR24 + CONST103 *
        VAR13 + CONST141 * VAR22) + x * (-CONST062 * VAR13 * VAR26 - 
        CONST092 * VAR17 * VAR22 + CONST112 * VAR15 * VAR24 + CONST144 * VAR20)
    Y07 = CONST132 * VAR03 * y * z + VAR05 * (CONST081 * VAR16 * z + 
        CONST107 * VAR25 * y) + VAR07 * (CONST026 * VAR14 * z + CONST044 *
        VAR16 * VAR25 + CONST107 * VAR23 * y) + x * (CONST025 * VAR14 *
        VAR25 + CONST053 * VAR12 * z + CONST081 * VAR16 * VAR23 + CONST132 *
        VAR21 * y)
    Y08 = CONST004 * VAR01 + VAR03 * (CONST006 * VAR26 + CONST116 * VAR17
        ) + VAR05 * (CONST008 * VAR24 + CONST069 * VAR15 + CONST087 * VAR17 *
        VAR26) + VAR07 * (CONST005 * VAR22 + CONST083 * VAR13 + CONST087 *
        VAR17 * VAR24 + CONST145 * VAR15 * VAR26) + x * (CONST004 * VAR20 +
        CONST022 * VAR11 + CONST069 * VAR15 * VAR24 + CONST082 * VAR13 *
        VAR26 + CONST116 * VAR17 * VAR22)
    Y09 = CONST009 * VAR10 + VAR12 * (CONST110 * VAR26 + CONST113 * VAR08
        ) + VAR14 * (CONST063 * VAR06 + CONST063 * VAR24 + CONST104 * VAR08 *
        VAR26) + VAR16 * (CONST056 * VAR06 * VAR26 + CONST056 * VAR08 *
        VAR24 + CONST101 * VAR04 + CONST101 * VAR22) + y * (CONST010 *
        VAR20 + CONST011 * VAR02 + CONST029 * VAR04 * VAR26 + CONST029 *
        VAR08 * VAR22 + CONST039 * VAR06 * VAR24)
    Y10 = CONST004 * VAR19 + VAR21 * (CONST005 * VAR08 + CONST117 * VAR17
        ) + VAR23 * (CONST008 * VAR06 + CONST070 * VAR15 + CONST088 * VAR08 *
        VAR17) + VAR25 * (CONST012 * VAR04 + CONST082 * VAR13 + CONST087 *
        VAR06 * VAR17 + CONST145 * VAR08 * VAR15) + z * (CONST004 * VAR02 +
        CONST023 * VAR11 + CONST070 * VAR06 * VAR15 + CONST084 * VAR08 *
        VAR13 + CONST117 * VAR04 * VAR17)
    Y11 = VAR12 * (CONST115 * VAR08 - CONST115 * VAR26) + VAR14 * (CONST066 *
        VAR06 + CONST072 * VAR24) + VAR16 * (CONST055 * VAR08 * VAR24 + 
        CONST093 * VAR04 + CONST093 * VAR06 * VAR26 - CONST093 * VAR22) + y * (
        CONST013 * VAR02 + CONST024 * VAR04 * VAR26 + CONST133 * VAR08 *
        VAR22 + CONST138 * VAR20)
    Y12 = CONST036 * VAR17 * VAR21 + CONST137 * VAR19 + VAR23 * (CONST086 *
        VAR15 + CONST123 * VAR08 * VAR17 - CONST143 * VAR06) + VAR25 * (
        CONST014 * VAR04 - CONST051 * VAR08 * VAR15 + CONST065 * VAR06 *
        VAR17 - CONST103 * VAR13) + z * (CONST062 * VAR08 * VAR13 + 
        CONST092 * VAR04 * VAR17 - CONST112 * VAR06 * VAR15 - CONST144 * VAR02)
    Y13 = VAR14 * (CONST049 * VAR06 + CONST049 * VAR24 + CONST090 * VAR08 *
        VAR26) + VAR16 * (CONST040 * VAR06 * VAR26 + CONST040 * VAR08 *
        VAR24 + CONST100 * VAR22 + CONST102 * VAR04) + y * (CONST016 *
        VAR20 + CONST017 * VAR02 + CONST094 * VAR06 * VAR24 + CONST121 *
        VAR04 * VAR26 + CONST122 * VAR08 * VAR22)
    Y14 = (CONST007 * VAR02 * z + CONST075 * VAR19 + CONST136 * VAR06 *
        VAR23 + CONST140 * VAR08 * VAR21 + VAR15 * (CONST032 * VAR08 *
        VAR25 + CONST047 * VAR23 + CONST131 * VAR06 * z) + VAR17 * (
        CONST068 * VAR06 * VAR25 + CONST073 * VAR04 * z + CONST114 * VAR08 *
        VAR23 + CONST128 * VAR21))
    Y15 = VAR16 * (CONST037 * VAR22 - CONST045 * VAR06 * VAR26 + CONST045 *
        VAR08 * VAR24 + CONST118 * VAR04) + y * (CONST018 * VAR02 + 
        CONST089 * VAR04 * VAR26 - CONST089 * VAR08 * VAR22 + CONST142 * VAR20)
    Y16 = (CONST019 * VAR02 * z + CONST076 * VAR19 + CONST124 * VAR04 *
        VAR25 - CONST129 * VAR08 * VAR21 + CONST134 * VAR06 * VAR23 + VAR17 *
        (CONST038 * VAR06 * VAR25 + CONST080 * VAR04 * z + CONST097 * VAR08 *
        VAR23 - CONST130 * VAR21))
    Y17 = y * (CONST058 * VAR02 + CONST058 * VAR20 + CONST061 * VAR04 *
        VAR26 + CONST061 * VAR08 * VAR22 + CONST074 * VAR06 * VAR24)
    Y18 = (CONST001 * VAR19 + CONST020 * VAR02 * z + CONST078 * VAR04 *
        VAR25 + CONST091 * VAR06 * VAR23 + CONST105 * VAR08 * VAR21)
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
    tl.store(output_ptr + output_row_offset + 17, Y17, mask=
        output_row_offset + 17 < output_numel)
    tl.store(output_ptr + output_row_offset + 18, Y18, mask=
        output_row_offset + 18 < output_numel)