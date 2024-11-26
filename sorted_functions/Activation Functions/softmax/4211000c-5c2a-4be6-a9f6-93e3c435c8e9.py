import triton
import triton.language as tl
import torch

@triton.jit
def softmax(input, log: 'tl.constexpr'):
    """
    Normalizes the input using softmax along the last dimension.

    Args:
        input: Input to normalize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        log: Flag for indicating if the log of softmax should be taken.

    Returns:
        Input normalized by softmax.
    """
    input = input
    input = input - tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    denominator = tl.sum(numerator, axis=1)[:, None]
    if log:
        output = input - tl.log(denominator)
    else:
        output = numerator / denominator
    return output
