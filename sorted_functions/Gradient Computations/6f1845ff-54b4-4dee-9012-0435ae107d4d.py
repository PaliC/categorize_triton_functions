import triton
import triton.language as tl
import torch

@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def dropout_backward_kernel(output_grad_pointer, input_grad_pointer, size,
    drop_p, seed, BLOCK_SIZE: 'tl.constexpr'):
    """
    Calculates the input gradient of dropout.

    Args:
        output_grad_pointer: Pointer to dropout's output gradients.
            The output gradients must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element used in dropout.
        seed: Seed for generating the dropout mask.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)
    tl.store(input_grad_pointer + offset, input_grad, mask=mask)
