import triton
import triton.language as tl
import torch

@triton.jit
def tl_softcapping_grad(dv: 'tl.tensor', v: 'tl.tensor', softcap: 'float'
    ) ->tl.tensor:
    v = v / softcap
    return dv * (1 - v * v)
