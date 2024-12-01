import triton
import triton.language as tl
import torch

@triton.jit
def liger_cross_entropy_kernel(X_ptr, X_stride, Y_ptr, Y_stride, loss_ptr,
    loss_stride, n_cols, n_non_ignore, ignore_index, label_smoothing:
    'tl.constexpr', reduction: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    X_ptr += program_id * X_stride
    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return
    loss_ptr += program_id * loss_stride
    m = float('-inf')
    d = 0.0
    ori_X_y = tl.load(X_ptr + y)
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other
            =float('-inf'))
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps *
                X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other
            =float('-inf'))
        if reduction == 'mean':
            X_block = (tl.exp(X_block - m) / d - eps) / n_non_ignore
        else:
            X_block = tl.exp(X_block - m) / d - eps
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)
    tl.debug_barrier()
    loss = -(ori_X_y - m - tl.log(d))
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss
    if reduction == 'mean':
        loss = loss / n_non_ignore
    X_y = tl.load(X_ptr + y)
    if reduction == 'mean':
        X_y += -(1 - label_smoothing) / n_non_ignore
    else:
        X_y += -(1 - label_smoothing)
    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)
