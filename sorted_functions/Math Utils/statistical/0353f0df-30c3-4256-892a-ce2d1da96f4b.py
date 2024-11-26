import triton
import triton.language as tl
import torch

@triton.jit
def clamp(x: 'tl.tensor', min_val, max_val) ->tl.tensor:
    """Clamps all elements in `x` into range [min, max].

    Args:
        x (tl.tensor): the input tensor.
        min_val (Number): lower bound of the range.
        max_val (Number): upper bound of the range.

    Returns:
        tl.tensor: the output tensor.
    """
    return tl.math.min(tl.math.max(x, min_val), max_val)
