import triton
import triton.language as tl
import torch

@triton.jit
def _rms_norm_backward_kernel(dY_ptr, dY_row_stride, X_ptr, X_row_stride,
    X_dtype: 'tl.constexpr', W_ptr, W_row_stride, RSTD_ptr, RSTD_row_stride,
    dW_ptr, dW_row_stride, n_rows, n_cols, offset, rows_per_program:
    'tl.constexpr', casting_mode: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    dY_ptr += row_start * dY_row_stride
    X_ptr += row_start * X_row_stride
    RSTD_ptr += row_start
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    W_row = W_row + offset
    for _ in range(row_start, row_end):
        dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(RSTD_ptr)
        X_row = X_row
        if casting_mode == _CASTING_MODE_LLAMA:
            m = dY_row * W_row
        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row
            m = dY_row * W_row
        else:
            m = dY_row * W_row
        dX_row = rstd_row * m
        dX_row += rstd_row * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(
            m * X_row, axis=0) * X_row)
        if casting_mode == _CASTING_MODE_LLAMA:
            dW_row += dY_row * (X_row * rstd_row)
        else:
            dW_row += dY_row * (X_row * rstd_row)
        tl.store(dY_ptr + col_offsets, dX_row, mask=mask)
        dY_ptr += dY_row_stride
        X_ptr += X_row_stride
        RSTD_ptr += RSTD_row_stride
    tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row,
        mask=mask)
