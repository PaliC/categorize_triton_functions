import triton
import triton.language as tl
import torch

@triton.jit
def _group_norm_backward_kernel(X_ptr, X_row_stride, X_col_stride, W_ptr,
    Mean_ptr, Mean_ptr_row_stride, Mean_ptr_col_stride, RSTD_ptr, DX_ptr,
    DW_ptr, DB_ptr, UPSTREAM_ptr, hidden_size: 'tl.constexpr',
    channels_per_group: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', dtype:
    'tl.constexpr'):
    """
    References:
    https://nn.labml.ai/normalization/group_norm/index.html
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md

    The backprop equations are the same for group_norm and layer_norm
    the only difference here is that we load the Mean, Rstd corresponding to the
    group we're computing gradients for and the mean and rstd are computed over n-channels
    so the total number of elements we compute the mean over is num_channels_per_group * hidden_size

    We also need to load the Weights corresponding to the current channel to compute the gradients.
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    X_ptr += batch_idx * X_row_stride
    DX_ptr += batch_idx * X_row_stride
    UPSTREAM_ptr += batch_idx * X_row_stride
    mean = tl.load(Mean_ptr + batch_idx * Mean_ptr_row_stride + group_idx *
        Mean_ptr_col_stride)
    rstd = tl.load(RSTD_ptr + batch_idx * Mean_ptr_row_stride + group_idx *
        Mean_ptr_col_stride)
    c1 = 0.0
    c2 = 0.0
    block_range = tl.arange(0, BLOCK_SIZE)
    for channel_idx in range(group_idx * channels_per_group, (group_idx + 1
        ) * channels_per_group):
        dW = 0.0
        dB = 0.0
        W = tl.load(W_ptr + channel_idx)
        for i in tl.range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(X_ptr + channel_idx * X_col_stride +
                hidden_size_offsets, mask=mask, other=0.0)
            UPSTREAM_grad = tl.load(UPSTREAM_ptr + channel_idx *
                X_col_stride + hidden_size_offsets, mask=mask, other=0.0)
            x_hat = (X - mean) * rstd
            dW += tl.sum(UPSTREAM_grad * x_hat)
            dB += tl.sum(UPSTREAM_grad)
            wdy = W * UPSTREAM_grad
            c1 += tl.sum(x_hat * wdy)
            c2 += tl.sum(wdy)
        tl.atomic_add(DW_ptr + channel_idx, dW)
        tl.atomic_add(DB_ptr + channel_idx, dB)
    N = hidden_size * channels_per_group
    c1 = c1 / N
    c2 = c2 / N
    for channel_idx in tl.range(group_idx * channels_per_group, (group_idx +
        1) * channels_per_group):
        W = tl.load(W_ptr + channel_idx)
        for i in range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            X = tl.load(X_ptr + channel_idx * X_col_stride +
                hidden_size_offsets, mask=mask, other=0.0)
            UPSTREAM_grad = tl.load(UPSTREAM_ptr + channel_idx *
                X_col_stride + hidden_size_offsets, mask=mask, other=0.0)
            x_hat = (X - mean) * rstd
            wdy = W * UPSTREAM_grad
            dx = (wdy - (x_hat * c1 + c2)) * rstd
            tl.store(DX_ptr + channel_idx * X_col_stride +
                hidden_size_offsets, dx, mask=mask)
