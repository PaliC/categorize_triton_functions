import triton
import triton.language as tl
import torch

@triton.jit
def triton_silu(x_ptr, b_ptr, xnumel, XBLOCK: 'tl.constexpr'):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x = tl.load(x_ptr + x0, mask=xmask)
    output = x * tl.sigmoid(x)
    tl.store(b_ptr + x0, output, xmask)
