import triton
import triton.language as tl
import torch

@triton.jit
def tl_logaddexp(a, b) ->tl.tensor:
    minx = tl.minimum(a, b)
    mx = tl.maximum(a, b)
    return tl_log1p(tl.exp(minx - mx)) + mx
