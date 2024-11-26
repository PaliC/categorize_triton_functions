import triton
import triton.language as tl
import torch

@triton.jit
def _group_norm_forward_kernel(Y_ptr, Y_row_stride, Y_col_stride, X_ptr,
    X_row_stride, X_col_stride, Mean_ptr, Mean_row_stride, Mean_col_stride,
    RSTD_ptr, RSTD_row_stride, RSTD_col_stride, W_ptr, B_ptr, hidden_size,
    channels_per_group, eps, BLOCK_SIZE: 'tl.constexpr'):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    X_ptr += batch_idx * X_row_stride + group_idx * X_col_stride
    Y_ptr += batch_idx * Y_row_stride + group_idx * Y_col_stride
    block_range = tl.arange(0, BLOCK_SIZE)
    s = 0.0
    squared_sum = 0.0
    for i in tl.range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + block_range
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0)
        s += tl.sum(X)
        squared_sum += tl.sum(X * X)
    m = s / hidden_size
    variance = squared_sum / hidden_size - m * m
    rstd = rsqrt(variance + eps)
    hidden_size_per_channel = hidden_size // channels_per_group
    for channel_idx in tl.range(group_idx * channels_per_group, (group_idx +
        1) * channels_per_group):
        W = tl.load(W_ptr + channel_idx)
        B = tl.load(B_ptr + channel_idx)
        for i in range(0, hidden_size_per_channel, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size_per_channel
            X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=m)
            Y = (X - m) * rstd * W + B
            tl.store(Y_ptr + hidden_size_offsets, Y, mask=mask)
        X_ptr += hidden_size_per_channel
        Y_ptr += hidden_size_per_channel
    tl.store(Mean_ptr + batch_idx * Mean_row_stride + group_idx *
        Mean_col_stride, m)
    tl.store(RSTD_ptr + batch_idx * RSTD_row_stride + group_idx *
        RSTD_col_stride, rstd)
