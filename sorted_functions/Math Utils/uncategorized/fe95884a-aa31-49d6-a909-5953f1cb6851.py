import triton
import triton.language as tl
import torch

@triton.jit
def elementwise_mul_kernel(x, g, N: 'tl.constexpr', B: 'tl.constexpr'):
    """
    This function multiplies each element of the tensor pointed by x with the value pointed by g.
    The multiplication is performed in-place on the tensor pointed by x.

    Parameters:
    x:
        Pointer to the input tensor.
    g:
        Pointer to the gradient output value.
    N (int):
        The number of columns in the input tensor.
    B (int):
        The block size for Triton operations.
    """
    i_x = tl.program_id(0)
    o_x = i_x * B + tl.arange(0, B)
    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)
