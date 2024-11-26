import triton
import triton.language as tl
import torch

@triton.jit
def _triton_second_order_bwd(x_ptr: 'tl.tensor', y_ptr: 'tl.tensor', z_ptr:
    'tl.tensor', g_x_ptr: 'tl.tensor', g_y_ptr: 'tl.tensor', g_z_ptr:
    'tl.tensor', g_1_0_ptr: 'tl.tensor', g_1_1_ptr: 'tl.tensor', g_1_2_ptr:
    'tl.tensor', g_2_0_ptr: 'tl.tensor', g_2_1_ptr: 'tl.tensor', g_2_2_ptr:
    'tl.tensor', g_2_3_ptr: 'tl.tensor', g_2_4_ptr: 'tl.tensor', BLOCK_SIZE:
    'tl.constexpr', vector_length: 'tl.constexpr'):
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
    tl.store(g_x_ptr + offset, g_x, mask=offset < vector_length)
    tl.store(g_y_ptr + offset, g_y, mask=offset < vector_length)
    tl.store(g_z_ptr + offset, g_z, mask=offset < vector_length)
