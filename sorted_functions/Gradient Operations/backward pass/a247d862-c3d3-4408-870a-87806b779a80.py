import triton
import triton.language as tl
import torch

@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def act_func_backward_kernel(output_grad_pointer, input_pointer,
    input_grad_pointer, size, drop_p, seed, param, act_func: 'tl.constexpr',
    dropout: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    Calculates the input gradient of an activation function.

    Args:
        output_grad_pointer: Pointer to the activation's output gradients.
            The output gradients must be of shape [size].
        input_pointer: Pointer to the activation's input.
            The input must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function whose gradient is calculated.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input = tl.load(input_pointer + offset, mask=mask)
    tl.store(input_grad_pointer + offset, apply_act_func_grad(output_grad,
        input, drop_p, seed, offset, param, act_func, dropout), mask=mask)
