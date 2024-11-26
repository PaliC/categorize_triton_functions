import triton
import triton.language as tl
import torch

@triton.jit
def _rms_norm_bwd_dx_fused(DX, DY, DW, DB, X, W, B, Rstd, Lock, stride, N,
    eps, GROUP_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    """
    Kernel invocation for backward pass of RMS normalization, computing gradients w.r.t. input

    Params:
        - DX (tensor): Gradient of the loss with respect to the inputs
        - DY (tensor): Gradient of the loss with respect to the outputs
        - DW (tensor): Gradient of the loss with respect to the scale tensor W
        - DB (tensor): Gradient of the loss with respect to the bias tensor B
        - X (tensor): Input tensor from the forward pass
        - W (tensor): Scale tensor applied during the forward pass
        - B (tensor): Bias tensor added during the forward pass
        - Rstd (tensor): Reciprocal of the standard deviation used for normalization in the forward pass
        - Lock (tensor): Lock tensor for atomic operations to prevent race conditions
        - stride (int): Stride to be applied when accessing elements in the tensors
        - N (int): Number of elements in each tensor
        - eps (float): Small epsilon value used during the forward pass
        - GROUP_SIZE_M (constexpr): Size of the group for M dimension, provided as a compile-time constant
        - BLOCK_SIZE_N (constexpr): Size of the block for N dimension, provided as a compile-time constant

    Return:
        - None

    Usage:
        _rms_norm_bwd_dx_fused[grid, block](DX, DY, DW, DB, X, W, B, Rstd, Lock, stride, N, eps, GROUP_SIZE_M, BLOCK_SIZE_N)
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0)
    dy = tl.load(DY + cols, mask=mask, other=0)
    w = tl.load(W + cols, mask=mask)
    rstd = tl.load(Rstd + row)
    x_norm = x * rstd
    wdy = w * dy
    dx = wdy * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = dy * x_norm
    partial_db = dy
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.atomic_xchg(Lock, 0)
