import triton
import triton.language as tl
import torch

@triton.jit
def dequantize(x: 'tl.tensor', scale: 'tl.tensor') ->tl.tensor:
    """Dequantize quantized tensor to floating point.

    Args:
        x (tl.tensor): quantized tensor.
        scale (tl.tensor): quantization scaling factor

    Returns:
        tl.tensor: Dequantized floating-point tensor.
    """
    return x * scale
