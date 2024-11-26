import triton
import triton.language as tl
import torch

@triton.heuristics({'GEMMA': lambda args: bool(args['GEMMA'])})
@triton.jit
def _rms_layernorm_backward(dY, dY_row_stride, dX, dX_row_stride, X,
    X_row_stride, W, W_row_stride, r, r_row_stride, n_cols, eps, GEMMA:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride
    if GEMMA:
        dX += row_idx * dY_row_stride
    else:
        dX = dY
    dY_row = tl.load(dY + col_offsets, mask=mask, other=0)
    X_row = tl.load(X + col_offsets, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    inv_var = tl.load(r)
    normed = X_row * inv_var
    if GEMMA:
        dY_W = dY_row * (W_row + 1.0)
    else:
        dY_W = dY_row * W_row
    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    tl.store(dX + col_offsets, output, mask=mask)
