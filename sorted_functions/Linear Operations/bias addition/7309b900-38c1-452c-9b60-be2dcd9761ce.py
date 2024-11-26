import triton
import triton.language as tl
import torch

@triton.jit
def add_vec_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: 'tl.constexpr',
    B1: 'tl.constexpr'):
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    off_x = block_id_x * B0 + tl.arange(0, B0)
    off_y = block_id_y * B1 + tl.arange(0, B1)
    off_z = off_y[:, None] * N0 + off_x[None, :]
    mask_x = off_x < N0
    mask_y = off_y < N1
    mask_z = mask_y[:, None] & mask_x[None, :]
    x = tl.load(x_ptr + off_x, mask=mask_x)
    y = tl.load(y_ptr + off_y, mask=mask_y)
    z = y[:, None] + x[None, :]
    tl.store(z_ptr + off_z, z, mask=mask_z)
    return
