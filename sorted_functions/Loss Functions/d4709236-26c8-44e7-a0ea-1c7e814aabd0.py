import triton
import triton.language as tl
import torch

@triton.jit
def nll_loss(input, size, reduction: 'tl.constexpr'):
    """
    Measures the negative log likelihood loss given log-probabilities of target class.

    Args:
        input: Input containing predicted log-probabilities corresponding to target class.
            The input can have arbitrary shape.
        size: Number of elements in the input.
            This value is used only if reduction is 'mean'.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.

    Returns:
        Loss.
    """
    input = input
    if reduction == 'none':
        output = -input
    elif reduction == 'mean':
        output = -tl.sum(input) / size
    elif reduction == 'sum':
        output = -tl.sum(input)
    return output
