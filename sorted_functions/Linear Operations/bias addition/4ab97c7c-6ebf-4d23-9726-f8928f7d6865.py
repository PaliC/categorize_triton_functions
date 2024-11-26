import triton
import triton.language as tl
import torch

@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: 'tl.constexpr', B1:
    'tl.constexpr'):
    off_x = tl.arange(0, B0)
    off_y = tl.arange(0, B1)
    off_z = off_y[:, None] * B0 + off_x[None, :]
    x = tl.load(x_ptr + off_x)
    y = tl.load(y_ptr + off_y)
    z = y[:, None] + x[None, :]
    tl.store(z_ptr + off_z, z)
    return
