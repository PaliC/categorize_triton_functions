import triton
import triton.language as tl
import torch

@triton.jit
def standardize(input, mean, inv_std, weight, bias):
    """
    Standardizes the input given its mean and inverse standard deviation,
    multiplies the result by weights, and adds a bias vector.

    Args:
        input: Input to standardize.
        mean: Mean of input.
        inv_std: Inverse standard deviation of input.
        weight: Weight multiplied by the standardized input.
        bias: Bias added to the result of the weight multiplication.

    Returns:
        Standardized input.
    """
    return weight * inv_std * (input - mean) + bias
