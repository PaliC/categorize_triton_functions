import triton
import triton.language as tl
import torch

@triton.jit
def quantize(x, scale, qmin, qmax) ->tl.tensor:
    """Quantize the tensor given quantization scale and data type.

    Args:
        x (tl.tensor): floating-point tensor
        scale (tl.tensor): quantization scale factor.
        qmin (Number): quantization minimum range.
        qmax (Number): quantization maximum range

    Returns:
        tl.tensor: rounded and clamped tensor.
            Note: this is still in floating point as we can't pass dtype to function

    Example:
    
        out = quantize(out, scale, -128, 127).to(tl.int8)
    """
    return clamp(tl.math.round(x / scale), qmin, qmax)
