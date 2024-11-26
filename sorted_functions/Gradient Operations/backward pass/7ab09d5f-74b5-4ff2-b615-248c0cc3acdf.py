import triton
import triton.language as tl
import torch

@triton.jit
def mul_relu_block_back_kernel(x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0:
    'tl.constexpr', B1: 'tl.constexpr'):
    block_id_i = tl.program_id(0)
    block_id_j = tl.program_id(1)
    off_i = block_id_i * B0 + tl.arange(0, B0)
    off_j = block_id_j * B1 + tl.arange(0, B1)
    off_ji = off_j[:, None] * N0 + off_i[None, :]
    mask_i = off_i < N0
    mask_j = off_j < N1
    mask_ji = mask_j[:, None] & mask_i[None, :]
    x = tl.load(x_ptr + off_ji, mask=mask_ji)
    y = tl.load(y_ptr + off_j, mask=mask_j)
    dz = tl.load(dz_ptr + off_ji, mask=mask_ji)
    df = tl.where(x * y[:, None] > 0, 1.0, 0.0)
    dxy_x = y[:, None]
    dx = df * dxy_x * dz
    tl.store(dx_ptr + off_ji, dx, mask=mask_ji)
    return
