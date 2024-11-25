import triton
import triton.language as tl
import torch

@triton.jit
def glu(input1, input2, param, act_func: 'tl.constexpr'):
    """
    Applies the gated linear unit with an arbitrary activation function
    to the input.

    Args:
        input1: First half of input to gate.
            The first half must be of the same shape as the second half.
        input2: Second half of input to gate.
            The second half must be of the same shape as the first half.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        param: Parameter in the case of parameterized activation functions.

    Args:
        Input transformed by the gated linear unit
        with an arbitrary activation function.
    """
    return input1 * apply_act_func(input2, None, None, None, param,
        act_func, False)
