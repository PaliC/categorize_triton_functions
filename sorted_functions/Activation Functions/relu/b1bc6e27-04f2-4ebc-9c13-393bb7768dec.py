import triton
import triton.language as tl
import torch

@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def act_func_forward_kernel(input_pointer, output_pointer, size, drop_p,
    seed, param, act_func: 'tl.constexpr', dropout: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr'):
    """
    Applies an activation function to the input, optionally fusing dropout.

    Args:
        input_pointer: Pointer to the input to transform.
            The input must be of shape [size].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input = tl.load(input_pointer + offset, mask=mask)
    tl.store(output_pointer + offset, apply_act_func(input, drop_p, seed,
        offset, param, act_func, dropout), mask=mask)
