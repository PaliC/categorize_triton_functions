import triton
import triton.language as tl
import torch

@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def p_loss_forward_kernel(input_pointer, target_pointer, output_pointer,
    size, p_loss: 'tl.constexpr', reduction: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr'):
    """
    Measures the L1 or squared L2 norm of the difference between the input
    and target (i.e., mean absolute error or mean squared error).

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [size].
        target_pointer: Pointer to the target.
            The target must be of shape [size].
        output_pointer: Pointer to a container the error is written to.
            The container must be of shape [size] if reduction is 'none',
            and otherwise of shape [size/BLOCK_SIZE].
        size: Number of elements in the input and target.
        p_loss: p-norm used to compute the error.
            Options are 1 for MAE and 2 for MSE.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
            If a reduction method is specified, the reduced result of each
            program is written to a separate index in the output container,
            which should later be summed.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input = tl.load(input_pointer + offset, mask=mask)
    target = tl.load(target_pointer + offset, mask=mask)
    diff = input - target
    if p_loss == 1:
        error = tl.abs(diff)
    elif p_loss == 2:
        error = diff * diff
    if reduction == 'none':
        tl.store(output_pointer + offset, error, mask=mask)
    elif reduction == 'mean':
        tl.store(output_pointer + pid, tl.sum(error) / size)
    elif reduction == 'sum':
        tl.store(output_pointer + pid, tl.sum(error))
