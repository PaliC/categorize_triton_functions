import triton
import triton.language as tl
import torch

@triton.jit
def fused_layer_norm_kernel(x_ptr, w_ptr, b_ptr, z_ptr, H, eps=1e-05,
    BLOCK_SIZE: 'tl.constexpr'=512):
    row_id = tl.program_id(0)
    x_ptr += row_id * H
    z_ptr += row_id * H
    x_mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, H, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offset, mask=offset < H, other=0.0)
        x_mean += x
    x_mean = tl.sum(x_mean) / H
    x_var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, H, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offset, mask=offset < H, other=x_mean)
        x = x
        x_var += (x - x_mean) * (x - x_mean)
    x_var = tl.sum(x_var) / H
    rstd = 1 / tl.sqrt(x_var + eps)
    for i in range(0, H, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < H
        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        w = tl.load(w_ptr + offset, mask=mask, other=0.0)
        b = tl.load(b_ptr + offset, mask=mask, other=0.0)
        z = (x - x_mean) * rstd
        z = z * w + b
        tl.store(z_ptr + offset, z, mask=mask)
