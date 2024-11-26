import triton
import triton.language as tl
import torch

@triton.jit
def eighth_order_bwd(coord_ptr: 'tl.tensor', coord_grad_ptr: 'tl.tensor',
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
    g_13 = tl.load(sph_grad_ptr + output_row_offset + 13, mask=
        output_row_offset + 13 < output_numel)
    g_14 = tl.load(sph_grad_ptr + output_row_offset + 14, mask=
        output_row_offset + 14 < output_numel)
    g_15 = tl.load(sph_grad_ptr + output_row_offset + 15, mask=
        output_row_offset + 15 < output_numel)
    g_16 = tl.load(sph_grad_ptr + output_row_offset + 16, mask=
        output_row_offset + 16 < output_numel)
    CONST000 = 2.0
    CONST001 = 3.0
    CONST002 = 4.50964677801932
    CONST004 = 5.0
    CONST005 = 6.78376969317208
    CONST006 = 4.0
    CONST007 = 9.01929355603863
    CONST008 = 6.76447016702898
    CONST009 = 6.0
    CONST011 = 13.5675393863442
    CONST012 = 15.0965641786467
    CONST013 = 13.136713523081
    CONST015 = 13.136713523081
    CONST017 = 19.4042118494929
    CONST019 = -489.184589393411
    CONST020 = 24.738633753706
    CONST023 = 26.2734270461621
    CONST024 = 27.0578806681159
    CONST025 = 24.738633753706
    CONST026 = 32.9848450049413
    CONST027 = 33.9188484658604
    CONST028 = 550.332663067587
    CONST030 = -978.369178786822
    CONST031 = 48.5105296237322
    CONST033 = 51.744564931981
    CONST035 = 48.9184589393411
    CONST041 = 65.6835676154051
    CONST043 = -1467.55376818023
    CONST045 = -12.2296147348353
    CONST047 = 582.126355484786
    CONST048 = -437.890450769368
    CONST049 = -434.108258927137
    CONST050 = -434.108258927137
    CONST052 = -432.926090689854
    CONST054 = -1447.02752975712
    CONST055 = 91.9569946615672
    CONST056 = -420.374832738593
    CONST057 = 6.46807061649763
    CONST058 = 97.0210592474644
    CONST061 = 103.489129863962
    CONST062 = -407.026181590325
    CONST063 = 108.231522672464
    CONST065 = 110.066532613517
    CONST066 = 110.066532613517
    CONST067 = 620.934779183772
    CONST068 = -396.284809689477
    CONST070 = 132.094936563159
    CONST071 = 434.108258927137
    CONST073 = 649.389136034781
    CONST076 = -366.888442045058
    CONST077 = -366.888442045058
    CONST078 = -361.756882439281
    CONST080 = -6.78376969317208
    CONST082 = -350.312360615494
    CONST083 = -346.340872551883
    CONST084 = -346.340872551883
    CONST085 = 173.170436275942
    CONST086 = 173.170436275942
    CONST088 = 183.444221022529
    CONST089 = 183.444221022529
    CONST090 = -325.62094527226
    CONST091 = -13.5289403340579
    CONST092 = -13.5675393863442
    CONST093 = 194.042118494929
    CONST095 = 197.050702846215
    CONST096 = -11.3224231339851
    CONST097 = 203.513090795162
    CONST098 = -814.05236318065
    CONST102 = -814.05236318065
    CONST104 = 217.054129463568
    CONST105 = 216.463045344927
    CONST106 = 220.133065227035
    CONST107 = -291.063177742393
    CONST108 = 220.133065227035
    CONST109 = -792.569619378954
    CONST111 = -271.350787726883
    CONST112 = 244.592294696705
    CONST113 = 244.592294696706
    CONST114 = 244.592294696706
    CONST115 = -776.168473979715
    CONST116 = -262.734270461621
    CONST117 = -259.755654413913
    CONST118 = -258.722824659905
    CONST120 = 262.734270461621
    CONST121 = -244.215708954195
    CONST122 = 271.350787726883
    CONST124 = -236.460843415458
    CONST127 = -217.054129463568
    CONST128 = -216.463045344927
    CONST129 = -216.463045344927
    CONST130 = -216.463045344927
    CONST131 = -723.513764878561
    CONST133 = -210.187416369296
    CONST134 = -210.187416369296
    CONST135 = 814.05236318065
    CONST136 = -197.050702846215
    CONST137 = 317.027847751582
    CONST138 = -194.042118494929
    CONST139 = -13.136713523081
    CONST140 = 324.694568017391
    CONST142 = 324.694568017391
    CONST143 = -175.156180307747
    CONST146 = -162.81047263613
    CONST147 = -162.347284008695
    CONST148 = 865.852181379709
    CONST149 = -158.513923875791
    CONST151 = -144.702752975712
    CONST152 = -649.389136034782
    CONST153 = -129.877827206956
    CONST154 = -129.361412329953
    CONST155 = 388.084236989858
    CONST157 = -115.446957517294
    CONST158 = -108.231522672464
    CONST159 = -108.231522672464
    CONST160 = 407.026181590325
    CONST161 = -103.489129863962
    CONST162 = -97.0210592474644
    CONST163 = -94.7025823384056
    CONST165 = -91.9569946615672
    CONST167 = -87.5780901538735
    CONST168 = -85.6073031438469
    CONST169 = -85.6073031438469
    CONST170 = -81.1736420043477
    CONST171 = 432.926090689854
    CONST172 = -79.2569619378954
    CONST173 = -81.1736420043477
    CONST177 = -79.2569619378954
    CONST178 = -72.3513764878561
    CONST179 = -72.1543484483091
    CONST180 = -70.0624721230988
    CONST181 = -72.1543484483091
    CONST182 = -67.8376969317208
    CONST183 = -65.6835676154052
    CONST184 = -61.1480736741764
    CONST185 = -1085.27064731784
    CONST186 = -61.1480736741764
    CONST187 = -1085.40315090753
    CONST188 = -57.7234787586472
    CONST189 = -12.9361412329953
    CONST190 = -1085.27064731784
    CONST191 = -52.8379746252636
    CONST192 = -51.744564931981
    CONST193 = -1585.13923875791
    CONST194 = -48.5105296237322
    CONST195 = -47.4863878522046
    CONST197 = 978.369178786822
    CONST198 = -517.44564931981
    CONST199 = -40.7026181590325
    CONST200 = -40.5868210021738
    CONST201 = -39.4101405692431
    CONST202 = -40.7026181590325
    CONST203 = -36.0771742241545
    CONST204 = -1056.75949250527
    CONST205 = -29.1063177742393
    CONST206 = 485.105296237322
    CONST207 = -26.2734270461621
    CONST208 = -26.4189873126318
    CONST209 = -1050.93708184648
    CONST210 = -22.6382471577417
    CONST211 = -20.6718218536732
    CONST212 = -19.4042118494929
    CONST213 = -20.3513090795162
    CONST214 = -528.379746252636
    CONST215 = -15.0965641786467
    CONST216 = -13.5675393863442
    CONST217 = -525.468540923241
    CONST218 = -11.3224231339851
    CONST219 = -13.5289403340579
    CONST220 = -9.70210592474644
    CONST221 = -10.3359109268366
    CONST222 = -6.46807061649763
    CONST223 = -13.136713523081
    CONST224 = -12.2296147348353
    CONST225 = -3.23403530824881
    CONST226 = -1034.89129863962
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
    g_x = tl.load(coord_grad_ptr + coord_row_offset, mask=coord_row_offset <
        coord_numel)
    g_y = tl.load(coord_grad_ptr + coord_row_offset + 1, mask=
        coord_row_offset + 1 < coord_numel)
    g_z = tl.load(coord_grad_ptr + coord_row_offset + 2, mask=
        coord_row_offset + 2 < coord_numel)
    g_x += g_0 * (CONST049 * VAR08 * VAR23 - CONST131 * VAR06 * VAR25 + 
        CONST151 * VAR04 * z - CONST211 * VAR21) + g_1 * y * (CONST178 *
        VAR04 - CONST178 * VAR22 + CONST185 * VAR08 * VAR24 - CONST190 *
        VAR06 * VAR26) + g_10 * (CONST017 * VAR05 * VAR26 + CONST161 *
        VAR13 * x - CONST189 * VAR03 - CONST198 * VAR07 * VAR15 + CONST222 *
        VAR22 * x + VAR17 * (CONST058 * VAR24 * x + CONST107 * VAR05 + 
        CONST138 * VAR07 * VAR26)) + g_11 * (CONST056 * VAR14 * x * z + 
        VAR16 * (-CONST082 * VAR25 * x - CONST209 * VAR07 * z) + y * (
        CONST116 * VAR07 * VAR25 + CONST124 * VAR05 * z + CONST207 * VAR23 * x)
        ) + g_12 * (CONST011 * VAR03 + CONST182 * VAR07 * VAR24 + CONST199 *
        VAR05 * VAR26 + CONST216 * VAR22 * x + VAR15 * (CONST098 * VAR26 *
        x + CONST122 * VAR07) + VAR17 * (-CONST102 * VAR07 * VAR26 + 
        CONST121 * VAR05 + CONST160 * VAR24 * x)) + g_13 * (VAR16 * (-
        CONST030 * VAR07 * z + CONST030 * VAR25 * x) + y * (CONST076 *
        VAR05 * z + CONST106 * VAR23 * x + CONST112 * VAR07 * VAR25)
        ) + g_14 * (CONST012 * VAR03 + CONST149 * VAR05 * VAR26 - CONST191 *
        VAR22 * x + VAR17 * (CONST109 * VAR24 * x + CONST149 * VAR05 - 
        CONST193 * VAR07 * VAR26)) + g_15 * y * (CONST050 * VAR05 * z + 
        CONST050 * VAR23 * x - CONST054 * VAR07 * VAR25) + g_16 * (CONST050 *
        VAR05 * VAR26 - CONST131 * VAR07 * VAR24 + CONST151 * VAR22 * x - 
        CONST211 * VAR03) + g_2 * (CONST001 * VAR08 * (-CONST208 * VAR23 + 
        CONST214 * VAR17 * VAR25) + CONST004 * VAR06 * (-CONST149 * VAR17 *
        z - CONST208 * VAR25) - CONST149 * VAR17 * VAR23 + CONST172 * VAR04 *
        z + CONST218 * VAR21) + g_3 * (VAR16 * (CONST043 * VAR08 * VAR26 + 
        CONST113 * VAR06 + CONST114 * VAR24) + y * (CONST028 * VAR06 *
        VAR26 + CONST088 * VAR08 * VAR24 + CONST168 * VAR04 + CONST184 * VAR22)
        ) + g_4 * (CONST001 * VAR08 * (CONST005 * VAR23 + CONST111 * VAR15 *
        z) + CONST004 * VAR06 * (CONST080 * VAR25 - CONST146 * VAR17 * z) +
        CONST005 * VAR21 - CONST111 * VAR15 * VAR25 + CONST146 * VAR17 *
        VAR23 + CONST195 * VAR04 * z) + g_5 * (VAR14 * (CONST133 * VAR08 - 
        CONST134 * VAR26) + VAR16 * (-CONST048 * VAR06 + CONST116 * VAR24 +
        CONST217 * VAR08 * VAR26) + y * (CONST041 * VAR06 * VAR26 + 
        CONST095 * VAR08 * VAR24 + CONST165 * VAR04 - CONST201 * VAR22)
        ) + g_6 * (CONST001 * VAR08 * (CONST093 * VAR17 * VAR25 + CONST118 *
        VAR15 * z + CONST220 * VAR23) + CONST004 * VAR06 * (-CONST162 *
        VAR17 * z + CONST220 * VAR25) + CONST118 * VAR15 * VAR25 - CONST161 *
        VAR13 * z - CONST162 * VAR17 * VAR23 + CONST210 * VAR04 * z + 
        CONST225 * VAR21) + g_7 * (CONST001 * VAR08 * (-CONST128 * VAR16 *
        VAR26 + CONST153 * VAR14 + CONST200 * VAR24 * y) + CONST004 * VAR06 *
        (CONST063 * VAR16 + CONST200 * VAR26 * y) + CONST020 * VAR12 + 
        CONST153 * VAR14 * VAR26 - CONST158 * VAR16 * VAR24 + CONST163 *
        VAR04 * y + CONST219 * VAR22 * y) + g_8 * (CONST000 * x * (CONST002 *
        VAR22 - CONST128 * VAR15 * VAR26 + CONST158 * VAR17 * VAR24 + 
        CONST188 * VAR13) + CONST006 * VAR07 * (CONST008 * VAR24 - CONST158 *
        VAR15 + CONST159 * VAR17 * VAR26) + CONST007 * VAR03 + CONST009 *
        VAR05 * (CONST002 * VAR26 + CONST203 * VAR17)) + g_9 * (CONST173 *
        VAR23 * x * y + VAR25 * (CONST147 * VAR07 * y + CONST171 * VAR16 *
        x) + z * (CONST117 * VAR14 * x + CONST170 * VAR05 * y + CONST171 *
        VAR07 * VAR16))
    g_y += CONST000 * g_14 * y * (-CONST068 * VAR06 * VAR26 + CONST068 *
        VAR08 * VAR24 + CONST208 * VAR04 - CONST208 * VAR22) + g_1 * (
        CONST078 * VAR07 * VAR24 + CONST104 * VAR05 * VAR26 - CONST178 *
        VAR22 * x + CONST221 * VAR03) + g_10 * (CONST000 * y * (CONST031 *
        VAR08 * VAR24 + CONST031 * VAR22 + CONST194 * VAR04 + CONST194 *
        VAR06 * VAR26) + CONST006 * VAR16 * (-CONST154 * VAR06 + CONST154 *
        VAR24) + CONST009 * VAR14 * (CONST033 * VAR26 + CONST192 * VAR08)
        ) + g_11 * (CONST001 * VAR17 * (-CONST116 * VAR06 * z - CONST143 *
        VAR08 * VAR25 + CONST167 * VAR23) + CONST004 * VAR15 * (CONST134 *
        VAR08 * z - CONST180 * VAR25) + CONST013 * VAR21 + CONST183 * VAR06 *
        VAR25 + CONST201 * VAR04 * z + CONST223 * VAR08 * VAR23) + g_12 * (
        CONST000 * y * (CONST097 * VAR06 * VAR26 + CONST097 * VAR08 * VAR24 +
        CONST199 * VAR04 + CONST199 * VAR22) + CONST006 * VAR16 * (CONST062 *
        VAR08 * VAR26 - CONST182 * VAR06 - CONST182 * VAR24)) + g_13 * (
        CONST001 * VAR17 * (CONST019 * VAR08 * VAR25 + CONST035 * VAR23 + 
        CONST113 * VAR06 * z) + CONST065 * VAR08 * VAR23 - CONST184 * VAR06 *
        VAR25 + CONST186 * VAR04 * z + CONST224 * VAR21) + g_15 * (-
        CONST078 * VAR06 * VAR25 + CONST127 * VAR08 * VAR23 + CONST178 *
        VAR04 * z - CONST221 * VAR21) + g_2 * (CONST137 * VAR05 * y * z + 
        CONST137 * VAR23 * x * y + CONST204 * VAR07 * VAR25 * y) + g_3 * (
        CONST001 * VAR17 * (CONST019 * VAR07 * VAR26 + CONST035 * VAR05 + 
        CONST114 * VAR24 * x) + CONST045 * VAR03 + CONST066 * VAR05 * VAR26 +
        CONST184 * VAR22 * x - CONST186 * VAR07 * VAR24) + g_4 * (-CONST090 *
        VAR05 * y * z + CONST187 * VAR07 * VAR16 * z + x * (CONST090 *
        VAR23 * y - CONST187 * VAR16 * VAR25)) + g_5 * (CONST001 * VAR17 *
        (CONST116 * VAR24 * x + CONST143 * VAR07 * VAR26 - CONST167 * VAR05
        ) + CONST004 * VAR15 * (-CONST134 * VAR26 * x + CONST180 * VAR07) +
        CONST015 * VAR05 * VAR26 + CONST041 * VAR07 * VAR24 + CONST139 *
        VAR03 - CONST201 * VAR22 * x) + g_6 * (-CONST138 * VAR05 * y * z + 
        VAR07 * (CONST155 * VAR25 * y + CONST226 * VAR16 * z) + x * (
        CONST067 * VAR14 * z - CONST138 * VAR23 * y + CONST226 * VAR16 * VAR25)
        ) + g_7 * (CONST219 * VAR03 + VAR05 * (CONST142 * VAR17 + CONST200 *
        VAR26) + VAR07 * (CONST152 * VAR15 - CONST152 * VAR17 * VAR26 + 
        CONST200 * VAR24) + x * (CONST085 * VAR13 + CONST140 * VAR17 *
        VAR24 + CONST152 * VAR15 * VAR26 + CONST219 * VAR22)) + g_8 * (
        CONST026 * VAR12 - CONST052 * VAR16 * VAR24 + CONST084 * VAR14 *
        VAR26 + CONST179 * VAR04 * y + CONST181 * VAR22 * y + VAR06 * (-
        CONST052 * VAR16 + CONST129 * VAR26 * y) + VAR08 * (CONST083 *
        VAR14 + CONST128 * VAR24 * y + CONST148 * VAR16 * VAR26)) + g_9 * (
        CONST219 * VAR21 + VAR23 * (CONST142 * VAR17 + CONST200 * VAR08) + 
        VAR25 * (CONST073 * VAR08 * VAR17 + CONST152 * VAR15 + CONST200 *
        VAR06) + z * (CONST086 * VAR13 + CONST091 * VAR04 + CONST142 *
        VAR06 * VAR17 + CONST152 * VAR08 * VAR15))
    g_z += g_0 * (-CONST049 * VAR05 * VAR26 + CONST131 * VAR07 * VAR24 - 
        CONST151 * VAR22 * x + CONST211 * VAR03) + g_1 * y * (-CONST050 *
        VAR23 * x + CONST054 * VAR07 * VAR25 + CONST071 * VAR05 * z) + g_10 * (
        CONST057 * VAR04 * z + CONST061 * VAR13 * z + CONST189 * VAR21 + 
        CONST198 * VAR15 * VAR25 + CONST212 * VAR08 * VAR23 + VAR17 * (
        CONST093 * VAR08 * VAR25 - CONST107 * VAR23 + CONST162 * VAR06 * z)
        ) + g_11 * (VAR14 * (-CONST133 * VAR26 + CONST134 * VAR08) + VAR16 *
        (CONST048 * VAR24 - CONST116 * VAR06 - CONST217 * VAR08 * VAR26) + 
        y * (CONST055 * VAR22 + CONST136 * VAR06 * VAR26 + CONST183 * VAR08 *
        VAR24 + CONST201 * VAR04)) + g_12 * (CONST011 * VAR21 + CONST092 *
        VAR04 * z + CONST182 * VAR06 * VAR25 + CONST202 * VAR08 * VAR23 + 
        VAR15 * (CONST098 * VAR08 * z + CONST122 * VAR25) + VAR17 * (-
        CONST102 * VAR08 * VAR25 + CONST121 * VAR23 + CONST160 * VAR06 * z)
        ) + g_13 * (VAR16 * (CONST043 * VAR08 * VAR26 + CONST113 * VAR06 + 
        CONST113 * VAR24) + y * (CONST028 * VAR08 * VAR24 + CONST089 *
        VAR06 * VAR26 + CONST169 * VAR22 + CONST186 * VAR04)) + g_14 * (-
        CONST149 * VAR08 * VAR23 + CONST191 * VAR04 * z + CONST215 * VAR21 +
        VAR17 * (-CONST109 * VAR06 * z - CONST149 * VAR23 + CONST193 *
        VAR08 * VAR25)) + g_15 * y * (CONST178 * VAR04 - CONST178 * VAR22 -
        CONST185 * VAR06 * VAR26 + CONST190 * VAR08 * VAR24) + g_16 * (
        CONST050 * VAR08 * VAR23 - CONST131 * VAR06 * VAR25 + CONST151 *
        VAR04 * z - CONST211 * VAR21) + g_2 * (CONST096 * VAR03 + VAR05 * (
        -CONST149 * VAR17 - CONST177 * VAR26) + VAR07 * (CONST070 * VAR24 +
        CONST193 * VAR17 * VAR26) + x * (-CONST109 * VAR17 * VAR24 + 
        CONST177 * VAR22)) + g_3 * (VAR16 * (CONST030 * VAR07 * z + 
        CONST197 * VAR25 * x) + y * (CONST077 * VAR23 * x + CONST108 *
        VAR05 * z + CONST114 * VAR07 * VAR25)) + g_4 * (CONST080 * VAR03 + 
        VAR05 * (-CONST146 * VAR17 + CONST213 * VAR26) + VAR07 * (CONST027 *
        VAR24 + CONST111 * VAR15) + x * (CONST102 * VAR17 * VAR24 + 
        CONST135 * VAR15 * VAR26 - CONST195 * VAR22)) + g_5 * (-CONST056 *
        VAR14 * x * z + VAR16 * (CONST082 * VAR07 * z + CONST209 * VAR25 *
        x) + y * (CONST023 * VAR05 * z + CONST120 * VAR07 * VAR25 - 
        CONST124 * VAR23 * x)) + g_6 * (CONST225 * VAR03 + VAR05 * (-
        CONST162 * VAR17 + CONST205 * VAR26) + VAR07 * (CONST047 * VAR17 *
        VAR26 + CONST118 * VAR15 + CONST194 * VAR24) + x * (CONST115 *
        VAR15 * VAR26 - CONST161 * VAR13 + CONST206 * VAR17 * VAR24 + 
        CONST210 * VAR22)) + g_7 * (CONST173 * VAR05 * y * z + VAR07 * (-
        CONST052 * VAR16 * z + CONST147 * VAR25 * y) + x * (-CONST052 *
        VAR16 * VAR25 + CONST117 * VAR14 * z + CONST173 * VAR23 * y)) + g_8 * (
        CONST007 * VAR04 * z + CONST007 * VAR21 - CONST052 * VAR15 * VAR25 +
        CONST130 * VAR17 * VAR23 + CONST157 * VAR13 * z + VAR06 * (CONST024 *
        VAR25 + CONST129 * VAR17 * z) + VAR08 * (CONST024 * VAR23 - 
        CONST052 * VAR15 * z + CONST052 * VAR17 * VAR25)) + g_9 * (CONST001 *
        VAR26 * (CONST105 * VAR08 * VAR16 + CONST153 * VAR14 + CONST200 *
        VAR06 * y) + CONST004 * VAR24 * (CONST063 * VAR16 + CONST200 *
        VAR08 * y) + CONST025 * VAR12 + CONST063 * VAR06 * VAR16 + CONST091 *
        VAR04 * y + CONST153 * VAR08 * VAR14 + CONST163 * VAR22 * y)
    tl.store(coord_grad_ptr + coord_row_offset, g_x, mask=coord_row_offset <
        coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 1, g_y, mask=
        coord_row_offset + 1 < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 2, g_z, mask=
        coord_row_offset + 2 < coord_numel)
