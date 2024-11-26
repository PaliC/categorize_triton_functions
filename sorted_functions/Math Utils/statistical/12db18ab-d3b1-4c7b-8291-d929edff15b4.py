import triton
import triton.language as tl
import torch

@triton.jit
def calc_mean_and_inv_std(input, last_dim, eps, last_dim_mask: 'tl.constexpr'):
    """
    Calculates the mean and inverse standard deviation of the input
    along the last dimension.

    Args:
        input: Input whose mean and inverse standard deviation are calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        last_dim: Size of the last dimension of input.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Mean and inverse standard deviation of the input.
    """
    input = input
    mean = tl.sum(input, axis=1) / last_dim
    diff = tl.where(last_dim_mask[None, :], input - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.sum(diff * diff, axis=1) / last_dim + eps)
    return mean, inv_std
