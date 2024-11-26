import triton
import triton.language as tl
import torch

@triton.jit
def fourth_order_bwd(coord_ptr: 'tl.tensor', coord_grad_ptr: 'tl.tensor',
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
    CONST000 = 2.0
    CONST001 = 4.5
    CONST002 = 2.25
    CONST006 = 9.48683298050514
    CONST008 = 12.0
    CONST012 = 28.4604989415154
    CONST014 = 40.2492235949962
    CONST015 = -37.6497011940334
    CONST016 = -6.70820393249937
    CONST017 = -26.6223590239483
    CONST018 = -21.3453742061366
    CONST019 = -20.1246117974981
    CONST020 = -18.8248505970167
    CONST021 = -18.0
    CONST022 = -14.2302494707577
    CONST023 = -10.0623058987491
    CONST024 = -9.0
    CONST025 = -8.87411967464942
    CONST026 = -7.11512473537885
    CONST027 = -6.27495019900557
    CONST028 = -3.35410196624968
    VAR07 = x * x * x
    VAR08 = x * x
    VAR16 = y * y * y
    VAR17 = y * y
    VAR25 = z * z * z
    VAR26 = z * z
    g_x = tl.load(coord_grad_ptr + coord_row_offset, mask=coord_row_offset <
        coord_numel)
    g_y = tl.load(coord_grad_ptr + coord_row_offset + 1, mask=
        coord_row_offset + 1 < coord_numel)
    g_z = tl.load(coord_grad_ptr + coord_row_offset + 2, mask=
        coord_row_offset + 2 < coord_numel)
    g_x += CONST015 * g_7 * x * y * z + CONST022 * g_5 * x * y * z + g_0 * (
        CONST017 * VAR08 * z - CONST025 * VAR25) + g_1 * y * (CONST020 *
        VAR08 - CONST020 * VAR26) + g_2 * (-CONST019 * VAR17 * z + CONST023 *
        VAR08 * z + CONST028 * VAR25) + g_3 * (CONST006 * VAR16 + CONST018 *
        VAR08 * y + CONST026 * VAR26 * y) + g_4 * (CONST000 * x * (CONST002 *
        VAR26 + CONST024 * VAR17) + CONST001 * VAR07) + g_6 * (-CONST016 *
        VAR07 + CONST019 * VAR17 * x) + g_8 * (CONST017 * VAR26 * x - 
        CONST025 * VAR07)
    g_y += CONST000 * g_6 * y * (CONST023 * VAR08 - CONST023 * VAR26
        ) + CONST014 * g_2 * x * y * z + g_1 * (-CONST020 * VAR26 * x + 
        CONST027 * VAR07) + g_3 * (CONST026 * VAR07 + x * (CONST012 * VAR17 +
        CONST026 * VAR26)) + g_4 * (CONST008 * VAR16 + CONST021 * VAR08 * y +
        CONST021 * VAR26 * y) + g_5 * (CONST026 * VAR25 + z * (CONST012 *
        VAR17 + CONST026 * VAR08)) + g_7 * (CONST020 * VAR08 * z - CONST027 *
        VAR25)
    g_z += -CONST015 * g_1 * x * y * z + CONST022 * g_3 * x * y * z + g_0 * (
        -CONST017 * VAR26 * x + CONST025 * VAR07) + g_2 * (CONST028 * VAR07 +
        x * (-CONST019 * VAR17 + CONST023 * VAR26)) + g_4 * (CONST001 *
        VAR08 * z + CONST001 * VAR25 + CONST021 * VAR17 * z) + g_5 * (
        CONST006 * VAR16 + CONST018 * VAR26 * y + CONST026 * VAR08 * y
        ) + g_6 * (CONST016 * VAR25 - CONST019 * VAR17 * z) + g_7 * y * (
        CONST020 * VAR08 - CONST020 * VAR26) + g_8 * (CONST017 * VAR08 * z -
        CONST025 * VAR25)
    tl.store(coord_grad_ptr + coord_row_offset, g_x, mask=coord_row_offset <
        coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 1, g_y, mask=
        coord_row_offset + 1 < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 2, g_z, mask=
        coord_row_offset + 2 < coord_numel)
