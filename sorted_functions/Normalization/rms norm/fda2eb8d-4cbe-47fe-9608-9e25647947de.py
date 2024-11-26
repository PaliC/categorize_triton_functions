import triton
import triton.language as tl
import torch

@triton.jit
def rms_layer_norm_fwd_fused(X, Y, W, RMS, stride, N, eps, BLOCK_SIZE:
    'tl.constexpr'):
    row = tl.program_id(axis=0)
    Y += row * stride
    X += row * stride
    mean = 0
    mean_ = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        x = tl.load(X + offset, mask=mask, other=0.0)
        mean_ += x * x
    mean = tl.sum(mean_, axis=0) / N
    rms = 1 / tl.sqrt(mean + eps)
    tl.store(RMS + row, rms)
    for i in range(0, N, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        x = tl.load(X + offset, mask=mask, other=0.0)
        x_hat = x * rms
        w = tl.load(W + offset, mask=mask)
        y = x_hat * w
        tl.store(Y + offset, y, mask=mask)
