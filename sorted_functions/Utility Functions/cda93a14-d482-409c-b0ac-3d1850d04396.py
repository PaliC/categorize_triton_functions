import triton
import triton.language as tl
import torch

@triton.jit
def update_welford(input, prev_count, prev_mean, prev_var, curr_count, mask:
    'tl.constexpr'):
    """
    Updates count, mean, and variance (M2) statistics for Welford's algorithm.

    Args:
        input: Input used to update statistics.
            The input must be of the same shape as the mask.
        prev_count: Previous count statistic to update.
        prev_mean: Previous mean statistic to update.
        prev_var: Previous variance (M2) statistic to update.
        curr_count: Count of elements in current input.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        Updated count, mean, and variance (M2) statistics
    """
    input = input
    count = prev_count + curr_count
    mean = (tl.sum(input) - curr_count * prev_mean) / count
    deltas = tl.where(mask, (input - mean) * (input - prev_mean), 0.0)
    var = prev_var + tl.sum(deltas)
    return count, mean, var
