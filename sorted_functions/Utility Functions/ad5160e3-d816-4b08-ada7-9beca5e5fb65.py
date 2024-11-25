import triton
import triton.language as tl
import torch

@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def dropout_forward_kernel(input_pointer, output_pointer, size, drop_p,
    seed, BLOCK_SIZE: 'tl.constexpr'):
    """
    Randomly zeroes elements in the input.

    Args:
        input_pointer: Pointer to the input to perform dropout on.
            The input must be of shape [size].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input = tl.load(input_pointer + offset, mask=mask)
    output = apply_dropout(input, drop_p, seed, offset)
    tl.store(output_pointer + offset, output, mask=mask)
