import triton
import triton.language as tl
import torch

@triton.jit
def tl_softcapping(v: 'tl.tensor', softcap: 'float') ->tl.tensor:
    return tl_tanh(v / softcap) * softcap
