import triton
import triton.language as tl
import torch

@triton.jit
def _triton_fourth_order_fwd(x_ptr: 'tl.tensor', y_ptr: 'tl.tensor', z_ptr:
    'tl.tensor', sh_1_0_ptr: 'tl.tensor', sh_1_1_ptr: 'tl.tensor',
    sh_1_2_ptr: 'tl.tensor', sh_2_0_ptr: 'tl.tensor', sh_2_1_ptr:
    'tl.tensor', sh_2_2_ptr: 'tl.tensor', sh_2_3_ptr: 'tl.tensor',
    sh_2_4_ptr: 'tl.tensor', sh_3_0_ptr: 'tl.tensor', sh_3_1_ptr:
    'tl.tensor', sh_3_2_ptr: 'tl.tensor', sh_3_3_ptr: 'tl.tensor',
    sh_3_4_ptr: 'tl.tensor', sh_3_5_ptr: 'tl.tensor', sh_3_6_ptr:
    'tl.tensor', sh_4_0_ptr: 'tl.tensor', sh_4_1_ptr: 'tl.tensor',
    sh_4_2_ptr: 'tl.tensor', sh_4_3_ptr: 'tl.tensor', sh_4_4_ptr:
    'tl.tensor', sh_4_5_ptr: 'tl.tensor', sh_4_6_ptr: 'tl.tensor',
    sh_4_7_ptr: 'tl.tensor', sh_4_8_ptr: 'tl.tensor', BLOCK_SIZE:
    'tl.constexpr', vector_length: 'tl.constexpr'):
    sqrt_3 = 3 ** 0.5
    block_id = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * block_id
    x_row_start = x_ptr + offset
    y_row_start = y_ptr + offset
    z_row_start = z_ptr + offset
    x = tl.load(x_row_start, mask=offset < vector_length)
    y = tl.load(y_row_start, mask=offset < vector_length)
    z = tl.load(z_row_start, mask=offset < vector_length)
    sh_1_0 = x * sqrt_3
    sh_1_1 = y * sqrt_3
    sh_1_2 = z * sqrt_3
    sqrt_15 = 15 ** 0.5
    sqrt_5 = 5 ** 0.5
    sq_x = x * x
    sq_y = y * y
    sq_z = z * z
    sh_2_0 = sqrt_15 * x * z
    sh_2_1 = sqrt_15 * x * y
    sh_2_2 = sqrt_5 * (sq_y - 0.5 * (sq_x + sq_z))
    sh_2_3 = sqrt_15 * y * z
    sh_2_4 = 0.5 * sqrt_15 * (sq_z - sq_x)
    sqrt_42 = 42 ** 0.5
    sqrt_168 = 168 ** 0.5
    sqrt_7 = 7 ** 0.5
    sh_3_0 = 1 / 6 * sqrt_42 * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = sqrt_7 * sh_2_0 * y
    sh_3_2 = 1 / 8 * sqrt_168 * (4 * sq_y - (sq_x + sq_z)) * x
    sh_3_3 = 0.5 * sqrt_7 * y * (2 * sq_y - 3 * (sq_x + sq_z))
    sh_3_4 = 1 / 8 * sqrt_168 * z * (4 * sq_y - (sq_x + sq_z))
    sh_3_5 = sqrt_7 * (sh_2_4 * y)
    sh_3_6 = 1 / 6 * sqrt_42 * (sh_2_4 * z - sh_2_0 * x)
    sqrt_2 = 2 ** 0.5
    sqrt_210 = 210 ** 0.5
    sqrt_14 = 14 ** 0.5
    sqrt_21 = 21 ** 0.5
    sqrt_70 = 70 ** 0.5
    sqrt_105 = 105 ** 0.5
    sqrt_6 = 6 ** 0.5
    sh_4_0 = 3 / 4 * sqrt_2 * (sh_3_0 * z + sh_3_6 * x)
    sh_4_1 = (3 / 4 * sh_3_0 * y + 3 / 8 * sqrt_6 * sh_3_1 * z + 3 / 8 *
        sqrt_6 * sh_3_5 * x)
    sh_4_2 = (-3 / 56 * sqrt_14 * sh_3_0 * z + 3 / 14 * sqrt_21 * sh_3_1 *
        y + 3 / 56 * sqrt_210 * sh_3_2 * z + 3 / 56 * sqrt_210 * sh_3_4 * x +
        3 / 56 * sqrt_14 * sh_3_6 * x)
    sh_4_3 = (-3 / 56 * sqrt_42 * sh_3_1 * z + 3 / 28 * sqrt_105 * sh_3_2 *
        y + 3 / 28 * sqrt_70 * sh_3_3 * x + 3 / 56 * sqrt_42 * sh_3_5 * x)
    sh_4_4 = (-3 / 28 * sqrt_42 * sh_3_2 * x + 3 / 7 * sqrt_7 * sh_3_3 * y -
        3 / 28 * sqrt_42 * sh_3_4 * z)
    sh_4_5 = (-3 / 56 * sqrt_42 * sh_3_1 * x + 3 / 28 * sqrt_70 * sh_3_3 *
        z + 3 / 28 * sqrt_105 * sh_3_4 * y - 3 / 56 * sqrt_42 * sh_3_5 * z)
    sh_4_6 = (-3 / 56 * sqrt_14 * sh_3_0 * x - 3 / 56 * sqrt_210 * sh_3_2 *
        x + 3 / 56 * sqrt_210 * sh_3_4 * z + 3 / 14 * sqrt_21 * sh_3_5 * y -
        3 / 56 * sqrt_14 * sh_3_6 * z)
    sh_4_7 = (-3 / 8 * sqrt_6 * sh_3_1 * x + 3 / 8 * sqrt_6 * sh_3_5 * z + 
        3 / 4 * sh_3_6 * y)
    sh_4_8 = 3 / 4 * sqrt_2 * (-sh_3_0 * x + sh_3_6 * z)
    sh_1_0_start = sh_1_0_ptr + offset
    sh_1_1_start = sh_1_1_ptr + offset
    sh_1_2_start = sh_1_2_ptr + offset
    sh_2_0_start = sh_2_0_ptr + offset
    sh_2_1_start = sh_2_1_ptr + offset
    sh_2_2_start = sh_2_2_ptr + offset
    sh_2_3_start = sh_2_3_ptr + offset
    sh_2_4_start = sh_2_4_ptr + offset
    sh_3_0_start = sh_3_0_ptr + offset
    sh_3_1_start = sh_3_1_ptr + offset
    sh_3_2_start = sh_3_2_ptr + offset
    sh_3_3_start = sh_3_3_ptr + offset
    sh_3_4_start = sh_3_4_ptr + offset
    sh_3_5_start = sh_3_5_ptr + offset
    sh_3_6_start = sh_3_6_ptr + offset
    sh_4_0_start = sh_4_0_ptr + offset
    sh_4_1_start = sh_4_1_ptr + offset
    sh_4_2_start = sh_4_2_ptr + offset
    sh_4_3_start = sh_4_3_ptr + offset
    sh_4_4_start = sh_4_4_ptr + offset
    sh_4_5_start = sh_4_5_ptr + offset
    sh_4_6_start = sh_4_6_ptr + offset
    sh_4_7_start = sh_4_7_ptr + offset
    sh_4_8_start = sh_4_8_ptr + offset
    tl.store(sh_1_0_start, sh_1_0, mask=offset < vector_length)
    tl.store(sh_1_1_start, sh_1_1, mask=offset < vector_length)
    tl.store(sh_1_2_start, sh_1_2, mask=offset < vector_length)
    tl.store(sh_2_0_start, sh_2_0, mask=offset < vector_length)
    tl.store(sh_2_1_start, sh_2_1, mask=offset < vector_length)
    tl.store(sh_2_2_start, sh_2_2, mask=offset < vector_length)
    tl.store(sh_2_3_start, sh_2_3, mask=offset < vector_length)
    tl.store(sh_2_4_start, sh_2_4, mask=offset < vector_length)
    tl.store(sh_3_0_start, sh_3_0, mask=offset < vector_length)
    tl.store(sh_3_1_start, sh_3_1, mask=offset < vector_length)
    tl.store(sh_3_2_start, sh_3_2, mask=offset < vector_length)
    tl.store(sh_3_3_start, sh_3_3, mask=offset < vector_length)
    tl.store(sh_3_4_start, sh_3_4, mask=offset < vector_length)
    tl.store(sh_3_5_start, sh_3_5, mask=offset < vector_length)
    tl.store(sh_3_6_start, sh_3_6, mask=offset < vector_length)
    tl.store(sh_4_0_start, sh_4_0, mask=offset < vector_length)
    tl.store(sh_4_1_start, sh_4_1, mask=offset < vector_length)
    tl.store(sh_4_2_start, sh_4_2, mask=offset < vector_length)
    tl.store(sh_4_3_start, sh_4_3, mask=offset < vector_length)
    tl.store(sh_4_4_start, sh_4_4, mask=offset < vector_length)
    tl.store(sh_4_5_start, sh_4_5, mask=offset < vector_length)
    tl.store(sh_4_6_start, sh_4_6, mask=offset < vector_length)
    tl.store(sh_4_7_start, sh_4_7, mask=offset < vector_length)
    tl.store(sh_4_8_start, sh_4_8, mask=offset < vector_length)
