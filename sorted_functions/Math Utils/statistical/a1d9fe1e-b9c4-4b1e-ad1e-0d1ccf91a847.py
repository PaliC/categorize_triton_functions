import triton
import triton.language as tl
import torch

@triton.jit
def calc_p_loss(input, target, size, p_loss: 'tl.constexpr', reduction:
    'tl.constexpr'):
    """
    Measures the L1 or squared L2 norm of the difference between the input
    and target (i.e., mean absolute error or mean squared error).

    Args:
        input: Input.
            The input must be of shape [BLOCK_SIZE].
        target: Target.
            The target must be of shape [BLOCK_SIZE].
        size: Number of elements in the input and target.
            This value is used only if reduction is 'mean'.
        p_loss: p-norm used to compute the error.
            Options are 1 for MAE and 2 for MSE.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.

    Returns:
        Error.
    """
    input = input
    target = target
    diff = input - target
    if p_loss == 1:
        error = tl.abs(diff)
    elif p_loss == 2:
        error = diff * diff
    if reduction == 'none':
        output = error
    elif reduction == 'mean':
        output = tl.sum(error) / size
    elif reduction == 'sum':
        output = tl.sum(error)
    return output
