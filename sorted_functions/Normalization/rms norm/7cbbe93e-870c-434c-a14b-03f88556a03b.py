import triton
import triton.language as tl
import torch

@triton.jit
def _rms_norm_fwd_fused(X, Y, W, B, Rstd, stride, N, eps, BLOCK_SIZE:
    'tl.constexpr'):
    """
    Kernel invocation for forward pass of RMS normalization with fused operations

    Params:
        - X (tensor): Input tensor
        - Y (tensor): Output tensor where the normalized results will be written
        - W (tensor): Scale tensor applied to the normalized input
        - B (tensor): Bias tensor added to the scaled input
        - Rstd (tensor): Reciprocal of the standard deviation used for normalization
        - stride (int): Stride to be applied when accessing elements in the input and output tensors
        - N (int): Number of elements in the input tensor
        - eps (float): Small epsilon value added to the variance to prevent division by zero
        - BLOCK_SIZE (constexpr): Size of the block for computation, provided as a compile-time constant

    Return:
        - None

    Usage:
        _rms_norm_fwd_fused[grid, block](X, Y, W, B, Rstd, stride, N, eps, BLOCK_SIZE)
    """
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    _rms = 0
    _rms = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0)
        _rms += a * a
    rms = tl.sqrt(tl.sum(_rms) / N + eps)
    tl.store(Rstd + row, rms)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0, eviction_policy=
            'evict_first')
        x_hat = x / rms
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)
