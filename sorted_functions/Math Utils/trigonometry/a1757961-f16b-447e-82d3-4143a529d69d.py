import triton
import triton.language as tl
import torch

@triton.jit
def _triton_fourth_order_bwd(x_ptr: 'tl.tensor', y_ptr: 'tl.tensor', z_ptr:
    'tl.tensor', g_x_ptr: 'tl.tensor', g_y_ptr: 'tl.tensor', g_z_ptr:
    'tl.tensor', g_1_0_ptr: 'tl.tensor', g_1_1_ptr: 'tl.tensor', g_1_2_ptr:
    'tl.tensor', g_2_0_ptr: 'tl.tensor', g_2_1_ptr: 'tl.tensor', g_2_2_ptr:
    'tl.tensor', g_2_3_ptr: 'tl.tensor', g_2_4_ptr: 'tl.tensor', g_3_0_ptr:
    'tl.tensor', g_3_1_ptr: 'tl.tensor', g_3_2_ptr: 'tl.tensor', g_3_3_ptr:
    'tl.tensor', g_3_4_ptr: 'tl.tensor', g_3_5_ptr: 'tl.tensor', g_3_6_ptr:
    'tl.tensor', g_4_0_ptr: 'tl.tensor', g_4_1_ptr: 'tl.tensor', g_4_2_ptr:
    'tl.tensor', g_4_3_ptr: 'tl.tensor', g_4_4_ptr: 'tl.tensor', g_4_5_ptr:
    'tl.tensor', g_4_6_ptr: 'tl.tensor', g_4_7_ptr: 'tl.tensor', g_4_8_ptr:
    'tl.tensor', BLOCK_SIZE: 'tl.constexpr', vector_length: 'tl.constexpr'):
    sqrt_3 = 3 ** 0.5
    sqrt_5 = 5 ** 0.5
    sqrt_15 = 15 ** 0.5
    block_id = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * block_id
    x_row_start = x_ptr + offset
    y_row_start = y_ptr + offset
    z_row_start = z_ptr + offset
    x = tl.load(x_row_start, mask=offset < vector_length)
    y = tl.load(y_row_start, mask=offset < vector_length)
    z = tl.load(z_row_start, mask=offset < vector_length)
    g_1_0 = tl.load(g_1_0_ptr + offset, mask=offset < vector_length)
    g_1_1 = tl.load(g_1_1_ptr + offset, mask=offset < vector_length)
    g_1_2 = tl.load(g_1_2_ptr + offset, mask=offset < vector_length)
    g_x = sqrt_3 * g_1_0
    g_y = sqrt_3 * g_1_1
    g_z = sqrt_3 * g_1_2
    g_2_0 = tl.load(g_2_0_ptr + offset, mask=offset < vector_length)
    g_2_1 = tl.load(g_2_1_ptr + offset, mask=offset < vector_length)
    g_2_2 = tl.load(g_2_2_ptr + offset, mask=offset < vector_length)
    g_2_3 = tl.load(g_2_3_ptr + offset, mask=offset < vector_length)
    g_2_4 = tl.load(g_2_4_ptr + offset, mask=offset < vector_length)
    g_x += sqrt_15 * z * g_2_0
    g_z += sqrt_15 * x * g_2_0
    g_x += sqrt_15 * y * g_2_1
    g_y += sqrt_15 * x * g_2_1
    g_y += sqrt_15 * z * g_2_2
    g_z += sqrt_15 * y * g_2_2
    g_x += -1.0 * sqrt_5 * x * g_2_3
    g_y += 2.0 * sqrt_5 * y * g_2_3
    g_z += -1.0 * sqrt_5 * z * g_2_3
    g_x += -1.0 * sqrt_15 * x * g_2_4
    g_z += sqrt_15 * z * g_2_4
    g_3_0 = tl.load(g_3_0_ptr + offset, mask=offset < vector_length)
    g_3_1 = tl.load(g_3_1_ptr + offset, mask=offset < vector_length)
    g_3_2 = tl.load(g_3_2_ptr + offset, mask=offset < vector_length)
    g_3_3 = tl.load(g_3_3_ptr + offset, mask=offset < vector_length)
    g_3_4 = tl.load(g_3_4_ptr + offset, mask=offset < vector_length)
    g_3_5 = tl.load(g_3_5_ptr + offset, mask=offset < vector_length)
    g_3_6 = tl.load(g_3_6_ptr + offset, mask=offset < vector_length)
    sq_x = x * x
    sq_y = y * y
    sq_z = z * z
    cu_z = sq_z * z
    cu_x = sq_x * x
    cu_y = sq_y * y
    g_x += sqrt_15 * g_3_0 * (-1.62018517460196 * sq_x + 1.08012344973464 *
        sq_z + 0.540061724867322 * sq_z)
    g_x += 2.64575131106459 * sqrt_15 * g_3_1 * y * z
    g_x -= g_3_2 * (4.8605555238059 * sq_x - 6.48074069840786 * sq_y + 
        1.62018517460197 * sq_z)
    g_x -= 7.93725393319377 * g_3_3 * x * y
    g_x -= 3.24037034920393 * g_3_4 * x * z
    g_x -= 2.64575131106459 * sqrt_15 * g_3_5 * x * y
    g_x -= sqrt_15 * g_3_6 * z * (1.08012344973464 * x + 2.16024689946929 * x)
    g_y += 2.64575131106459 * sqrt_15 * g_3_1 * x * z
    g_y += 12.9614813968157 * g_3_2 * x * y
    g_y -= g_3_3 * (3.96862696659689 * sq_x - 7.93725393319377 * sq_y + 
        3.96862696659689 * sq_z)
    g_y += 12.9614813968157 * g_3_4 * y * z
    g_y -= 1.3228756555323 * sqrt_15 * g_3_5 * (sq_x - sq_z)
    g_z += sqrt_15 * g_3_0 * x * (1.08012344973464 * z + 2.16024689946929 * z)
    g_z += 2.64575131106459 * sqrt_15 * g_3_1 * x * y
    g_z -= 3.24037034920393 * g_3_2 * x * z
    g_z -= 7.93725393319377 * g_3_3 * y * z
    g_z -= g_3_4 * (1.62018517460197 * sq_x - 6.48074069840786 * sq_y + 
        4.8605555238059 * sq_z)
    g_z += 2.64575131106459 * sqrt_15 * g_3_5 * y * z
    g_z -= sqrt_15 * g_3_6 * (1.08012344973464 * sq_x + 0.540061724867322 *
        sq_x - 1.62018517460196 * sq_z)
    g_4_0 = tl.load(g_4_0_ptr + offset, mask=offset < vector_length)
    g_4_1 = tl.load(g_4_1_ptr + offset, mask=offset < vector_length)
    g_4_2 = tl.load(g_4_2_ptr + offset, mask=offset < vector_length)
    g_4_3 = tl.load(g_4_3_ptr + offset, mask=offset < vector_length)
    g_4_4 = tl.load(g_4_4_ptr + offset, mask=offset < vector_length)
    g_4_5 = tl.load(g_4_5_ptr + offset, mask=offset < vector_length)
    g_4_6 = tl.load(g_4_6_ptr + offset, mask=offset < vector_length)
    g_4_7 = tl.load(g_4_7_ptr + offset, mask=offset < vector_length)
    g_4_8 = tl.load(g_4_8_ptr + offset, mask=offset < vector_length)
    g_x -= sqrt_15 * g_4_0 * (3.43693177121688 * sq_x * z + 
        3.43693177121688 * sq_x * z - 1.14564392373896 * cu_z - 
        1.14564392373896 * cu_z)
    g_x += sqrt_15 * g_4_1 * y * (-4.8605555238059 * sq_x + 
        3.24037034920393 * sq_z + 1.62018517460197 * sq_z)
    g_x -= g_4_2 * (0.649519052838329 * sqrt_15 * sq_x * z + 
        7.54672942406179 * sq_x * z - 2.59807621135332 * sqrt_15 * sq_y * z -
        10.0623058987491 * sq_y * z + 0.21650635094611 * sqrt_15 * cu_z + 
        2.51557647468726 * cu_z)
    g_x -= g_4_3 * y * (0.918558653543692 * sqrt_15 * sq_x + 
        16.0090306546024 * sq_x - 9.48683298050514 * sq_y + 
        0.918558653543692 * sqrt_15 * sq_z + 5.33634355153414 * sq_z + 
        0.459279326771846 * sqrt_15 * (sq_x - sq_z))
    g_x += g_4_4 * (-9.0 * x * sq_y + 2.25 * x * sq_z - 9.0 * x * sq_y + 
        2.25 * x * sq_z + 4.5 * cu_x)
    g_x -= g_4_5 * y * z * (-0.918558653543692 * sqrt_15 * x + 
        10.6726871030683 * x + 1.83711730708738 * sqrt_15 * x)
    g_x -= g_4_6 * (2.59807621135332 * sqrt_15 * x * sq_y - 
        0.21650635094611 * sqrt_15 * x * sq_z + 2.51557647468726 * x * sq_z +
        10.0623058987491 * x * sq_y - 2.51557647468726 * x * sq_z + 
        0.21650635094611 * sqrt_15 * x * sq_z - 5.03115294937453 * cu_x - 
        0.433012701892219 * sqrt_15 * cu_x)
    g_x -= sqrt_15 * g_4_7 * y * z * (3.24037034920393 * x + 
        6.48074069840786 * x)
    g_x -= sqrt_15 * g_4_8 * (1.14564392373896 * x * sq_z + 
        4.58257569495584 * x * sq_z + 1.14564392373896 * x * sq_z - 
        2.29128784747792 * cu_x)
    g_y += sqrt_15 * g_4_1 * x * (-1.62018517460197 * sq_x + 
        3.24037034920393 * sq_z + 1.62018517460197 * sq_z)
    g_y += g_4_2 * x * z * (5.19615242270663 * sqrt_15 * y + 
        20.1246117974981 * y)
    g_y -= g_4_3 * x * (5.33634355153414 * sq_x - 28.4604989415154 * sq_y +
        0.918558653543692 * sqrt_15 * sq_z + 5.33634355153414 * sq_z + 
        0.459279326771846 * sqrt_15 * (sq_x - sq_z))
    g_y -= g_4_4 * (9.0 * sq_x * y + 9.0 * sq_x * y + 9.0 * y * sq_z + 9.0 *
        y * sq_z - 12.0 * cu_y)
    g_y -= g_4_5 * z * (0.918558653543692 * sqrt_15 * sq_x + 
        5.33634355153414 * sq_x - 28.4604989415154 * sq_y + 
        5.33634355153414 * sq_z - 0.459279326771846 * sqrt_15 * (sq_x - sq_z))
    g_y -= g_4_6 * (10.0623058987491 * sq_x * y + 2.59807621135332 *
        sqrt_15 * y * (sq_x - sq_z) - 10.0623058987491 * y * sq_z)
    g_y -= sqrt_15 * g_4_7 * z * (3.24037034920393 * sq_x + 
        1.62018517460197 * sq_x - 1.62018517460197 * sq_z)
    g_z -= sqrt_15 * g_4_0 * (1.14564392373896 * cu_x - 3.43693177121688 *
        x * sq_z - 3.43693177121688 * x * sq_z + 1.14564392373896 * cu_x)
    g_z += sqrt_15 * g_4_1 * x * y * (3.24037034920393 * z + 
        6.48074069840786 * z)
    g_z -= g_4_2 * (0.21650635094611 * sqrt_15 * cu_x - 2.59807621135332 *
        sqrt_15 * x * sq_y - 10.0623058987491 * x * sq_y + 
        0.649519052838329 * sqrt_15 * x * sq_z + 7.54672942406179 * x *
        sq_z + 2.51557647468726 * cu_x)
    g_z -= g_4_3 * x * y * (-0.918558653543692 * sqrt_15 * z + 
        10.6726871030683 * z + 1.83711730708738 * sqrt_15 * z)
    g_z += g_4_4 * (2.25 * sq_x * z + 2.25 * sq_x * z - 9.0 * sq_y * z - 
        9.0 * sq_y * z + 4.5 * cu_z)
    g_z -= g_4_5 * y * (0.918558653543692 * sqrt_15 * sq_x + 
        5.33634355153414 * sq_x - 9.48683298050514 * sq_y + 
        0.918558653543692 * sqrt_15 * sq_z + 16.0090306546024 * sq_z - 
        0.459279326771846 * sqrt_15 * (sq_x - sq_z))
    g_z += g_4_6 * (-0.21650635094611 * sqrt_15 * sq_x * z + 
        2.51557647468726 * sq_x * z - 2.51557647468726 * sq_x * z + 
        0.21650635094611 * sqrt_15 * sq_x * z + 2.59807621135332 * sqrt_15 *
        sq_y * z + 10.0623058987491 * sq_y * z - 5.03115294937453 * cu_z - 
        0.433012701892219 * sqrt_15 * cu_z)
    g_z -= sqrt_15 * g_4_7 * y * (3.24037034920393 * sq_x + 
        1.62018517460197 * sq_x - 4.8605555238059 * sq_z)
    g_z -= sqrt_15 * g_4_8 * (1.14564392373896 * sq_x * z + 
        4.58257569495584 * sq_x * z + 1.14564392373896 * sq_x * z - 
        2.29128784747792 * cu_z)
    tl.store(g_x_ptr + offset, g_x, mask=offset < vector_length)
    tl.store(g_y_ptr + offset, g_y, mask=offset < vector_length)
    tl.store(g_z_ptr + offset, g_z, mask=offset < vector_length)
