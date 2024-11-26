import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_loss(input, pred):
    """
    Measures the per-row cross entropy loss given
    input and predicted logits corresponding to target class.

    Args:
        input: Input.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        pred: Predicted logits corresponding to target class.
            The predictions must be of shape [BLOCK_SIZE1].

    Returns:
        Loss.
    """
    input = input
    pred = pred
    mx = tl.max(input, axis=1)
    input -= mx[:, None]
    loss = tl.log(tl.sum(tl.exp(input), axis=1)) - pred + mx
    return loss
