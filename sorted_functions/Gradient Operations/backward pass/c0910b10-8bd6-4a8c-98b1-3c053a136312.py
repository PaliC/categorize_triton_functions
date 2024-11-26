import triton
import triton.language as tl
import torch

@triton.jit
def dcomp2dx(x, b_return):
    _return2 = tl.expand_dims(x, 1)
    bx = zeroslike(x)
    b_return2 = zeroslike(_return2)
    _b_return2 = triton_unbroadcast(b_return * x, _return2.shape)
    return bx
