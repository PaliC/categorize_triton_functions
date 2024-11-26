import triton
import triton.language as tl
import torch

@triton.jit
def layer_norm_kernel(x, mean, var, gamma, beta, epsilon, stride_xm,
    stride_xn, stride_gamma, stide_beta, n, BLOCK_SIZE: 'tl.constexpr'):
    """
    Triton kernel for layer normalization.

    Parameters:
    x - Input tensor.
    mean - Tensor to store computed means.
    var - Tensor to store computed variances.
    gamma - Scale tensor.
    beta - Shift tensor.
    epsilon - A small value to avoid division by zero.
    stride_xm, stride_xn - Strides for the input tensor.
    stride_gamma, stride_beta - Strides for Gamma and Beta tensors.
    n - Size of the last dimension of the input tensor.
    BLOCK_SIZE - Size of the block for Triton computation.
    """
    row = tl.program_id(0)
    x_ptrs = x + row * stride_xm
    mean_ptrs = mean + row
    var_ptrs = var + row
    gamma_ptrs = gamma
    beta_ptrs = beta
    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK_SIZE) < n, other=0)
    mean = tl.sum(x, axis=0) / n
    tl.store(mean_ptrs, mean)
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n
    tl.store(var_ptrs, var)
    std = tl.sqrt(var + epsilon)
    y = x_centered / std * tl.load(gamma_ptrs, mask=tl.arange(0, BLOCK_SIZE
        ) < n, other=1) + tl.load(beta_ptrs, mask=tl.arange(0, BLOCK_SIZE) <
        n, other=0)
    tl.store(x_ptrs, y, mask=tl.arange(0, BLOCK_SIZE) < n)
