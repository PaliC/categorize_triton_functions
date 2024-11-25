import triton
import triton.language as tl
import torch

@triton.jit
def update_ema(prev_ema, new_val, momentum):
    """
    Updates exponential moving average.

    Args:
        prev_ema: Previous exponential moving average.
        new_val: Value used to update the exponential moving average.
        momentum: Momentum.

    Returns:
        Updated running statistic.
    """
    return (1 - momentum) * prev_ema + momentum * new_val
