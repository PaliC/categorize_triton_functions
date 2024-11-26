import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_N_COLS': 32,
    'BLOCK_SIZE_N_ROWS': 32}, num_stages=3, num_warps=1), triton.Config({
    'BLOCK_SIZE_N_COLS': 64, 'BLOCK_SIZE_N_ROWS': 32}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_SIZE_N_COLS': 128,
    'BLOCK_SIZE_N_ROWS': 32}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_N_COLS': 256, 'BLOCK_SIZE_N_ROWS': 32}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_SIZE_N_COLS': 32,
    'BLOCK_SIZE_N_ROWS': 64}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_N_COLS': 64, 'BLOCK_SIZE_N_ROWS': 64}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_SIZE_N_COLS': 128,
    'BLOCK_SIZE_N_ROWS': 64}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_N_COLS': 256, 'BLOCK_SIZE_N_ROWS': 64}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_SIZE_N_COLS': 32,
    'BLOCK_SIZE_N_ROWS': 128}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_N_COLS': 64, 'BLOCK_SIZE_N_ROWS': 128}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_SIZE_N_COLS': 128,
    'BLOCK_SIZE_N_ROWS': 128}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_N_COLS': 256, 'BLOCK_SIZE_N_ROWS': 128}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_SIZE_N_COLS': 32,
    'BLOCK_SIZE_N_ROWS': 256}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_N_COLS': 64, 'BLOCK_SIZE_N_ROWS': 256}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_SIZE_N_COLS': 128,
    'BLOCK_SIZE_N_ROWS': 256}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_N_COLS': 256, 'BLOCK_SIZE_N_ROWS': 256}, num_stages=3,
    num_warps=8)], key=['n_cols', 'n_rows'], reset_to_zero=[
    'per_channel_amax_ptr'])
@triton.jit
def fast_abs_max_kernel(weights_ptr, per_channel_amax_ptr, col_stride,
    row_stride, n_cols, n_rows, BLOCK_SIZE_N_COLS: 'tl.constexpr',
    BLOCK_SIZE_N_ROWS: 'tl.constexpr'):
    """
    Computes the per-channel absolute maximum of the weights.
    Args:
        weights_ptr (pointer): pointer to the weights
        per_channel_amax_ptr (pointer): pointer to the per-channel amax output vector
        col_stride (int): stride for moving to the next row of the matrix (next column)
        row_stride (int): stride for moving to the next column of the matrix (next row)
        n_cols (int): number of columns in the weight matrix (assuming x @ W^T).
            So n_cols is n_rows of W
        n_rows (int): number of rows. - same as above -
        BLOCK_SIZE_N_COLS (tl.constexpr): block size for operating along the columns
        BLOCK_SIZE_N_ROWS (tl.constexpr): block size for operating along the rows
    """
    pid = tl.program_id(0)
    n_pid_rows = tl.cdiv(n_rows, BLOCK_SIZE_N_ROWS)
    col_block_idx = pid // n_pid_rows
    row_block_idx = pid % n_pid_rows
    col_offs = col_block_idx * BLOCK_SIZE_N_COLS + tl.arange(0,
        BLOCK_SIZE_N_COLS)
    row_offs = row_block_idx * BLOCK_SIZE_N_ROWS + tl.arange(0,
        BLOCK_SIZE_N_ROWS)
    ptrs = weights_ptr + (col_offs[:, None] * col_stride + row_offs[None, :
        ] * row_stride)
    block_weights = tl.load(ptrs, mask=(col_offs[:, None] < n_cols) & (
        row_offs[None, :] < n_rows), other=float('-inf'))
    block_weights_abs = tl.where((col_offs[:, None] < n_cols) & (row_offs[
        None, :] < n_rows), tl.abs(block_weights), float('-inf'))
    abs_max_block_weights = tl.max(block_weights_abs, axis=1)
    tl.atomic_max(per_channel_amax_ptr + col_offs, abs_max_block_weights,
        mask=col_offs < n_cols)
