import triton
import triton.language as tl
import torch

@triton.jit
def srms_norm_fw(X, Y, V, stride, N, eps, BLOCK_SIZE_N: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x_zm = tl.where(mask, x, 0.0)
    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)
    y = x_zm * rstd
    tl.store(V + row, rstd)
    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=mask)
