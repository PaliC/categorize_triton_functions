import triton
import triton.language as tl
import torch

@triton.jit
def apply_act_func(input, drop_p, seed, offset, param, act_func:
    'tl.constexpr', dropout: 'tl.constexpr'):
    """
    Applies an activation function to the input, optionally fusing dropout.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Input transformed by the desired activation function,
        potentially with fused dropout.
    """
    if act_func == 'sigmoid':
        input = input
        output = sigmoid(input)
    elif act_func == 'tanh':
        input = input
        output = tanh(input)
    elif act_func == 'relu':
        output = relu(input)
    elif act_func == 'gelu':
        input = input
        output = gelu(input)
    elif act_func == 'silu':
        input = input
        output = silu(input)
    elif act_func == 'relu6':
        output = relu6(input)
    elif act_func == 'hardsigmoid':
        output = hardsigmoid(input)
    elif act_func == 'hardswish':
        output = hardswish(input)
    elif act_func == 'selu':
        input = input
        output = selu(input)
    elif act_func == 'mish':
        input = input
        output = mish(input)
    elif act_func == 'leaky_relu':
        output = leaky_relu(input, param)
    if dropout:
        output = apply_dropout(output, drop_p, seed, offset)
    return output
