import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_bwd_dx_fused(_DA, _DOut, _A, Weight, Mean, Rstd, stride,
    NumRows, NumCols, eps, BLOCK_SIZE_N: 'tl.constexpr'):
    pid = tl.program_id(0)
    row = pid
    A = _A + row * stride
    DOut = _DOut + row * stride
    DA = _DA + row * stride
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    _mean1 = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    _mean2 = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for off in range(0, NumCols, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < NumCols
        a = tl.load(A + cols, mask=mask, other=0)
        dout = tl.load(DOut + cols, mask=mask, other=0)
        weight = tl.load(Weight + cols, mask=mask, other=0)
        a_hat = (a - mean) * rstd
        wdout = weight * dout
        _mean1 += a_hat * wdout
        _mean2 += wdout
    mean1 = tl.sum(_mean1, axis=0) / NumCols
    mean2 = 0.0
    mean2 = tl.sum(_mean2, axis=0) / NumCols
    for off in range(0, NumCols, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < NumCols
        a = tl.load(A + cols, mask=mask, other=0)
        dout = tl.load(DOut + cols, mask=mask, other=0)
        weight = tl.load(Weight + cols, mask=mask, other=0)
        a_hat = (a - mean) * rstd
        wdout = weight * dout
        da = (wdout - (a_hat * mean1 + mean2)) * rstd
        tl.store(DA + cols, da, mask=mask)
