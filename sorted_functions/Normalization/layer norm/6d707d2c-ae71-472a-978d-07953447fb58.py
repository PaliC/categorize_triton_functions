import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_fused(X, Y, W, M, V, stride, N, BLOCK_SIZE: 'tl.constexpr'
    ):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    X += row * stride
    Y += row * stride
    x = tl.load(X + cols, mask=mask, other=0)
    mean = tl.sum(x, axis=0) / N
    xmean = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xmean * xmean, axis=0) / N
    rstd = 1 / tl.sqrt(var + 1e-05)
    xhat = xmean * rstd
    tl.store(M + row, mean)
    tl.store(V + row, rstd)
    w = tl.load(W + cols, mask=mask)
    y = xhat * w
    tl.store(Y + cols, y, mask=mask)
