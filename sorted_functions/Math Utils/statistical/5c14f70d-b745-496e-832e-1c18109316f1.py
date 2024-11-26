import triton
import triton.language as tl
import torch

@triton.jit
def _jsd_kernel(X_ptr, X_stride, Y_ptr, Y_stride, loss_ptr, loss_stride,
    dX_ptr, dX_stride, label_ptr, beta: 'tl.constexpr', n_non_ignore: 'int',
    ignore_index: 'tl.constexpr', n_cols, BLOCK_SIZE: 'tl.constexpr',
    HAS_LABEL: 'tl.constexpr'):
    pid = tl.program_id(0)
    X_ptr += pid * X_stride
    dX_ptr += pid * dX_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride
    label_ptr += pid
    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + offsets, 0.0, mask=offsets < n_cols)
            return
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float('-inf'))
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float('-inf'))
        if beta == 0.0:
            Y_prob = tl.exp(Y)
            loss = Y_prob * (Y - X)
            dX = -Y_prob
        elif beta == 1.0:
            X_prob = tl.exp(X)
            loss = X_prob * (X - Y)
            dX = loss + X_prob
        else:
            Q = tl.exp(X)
            P = tl.exp(Y)
            M = beta * P + (1 - beta) * Q
            log_M = tl.log(M)
            loss = beta * P * Y + (1 - beta) * Q * X - M * log_M
            dX = (1 - beta) * Q * (X - log_M)
        loss = loss / n_non_ignore
        dX = dX / n_non_ignore
        tl.store(loss_ptr + offsets, loss, mask=mask)
        tl.store(dX_ptr + offsets, dX, mask=mask)
