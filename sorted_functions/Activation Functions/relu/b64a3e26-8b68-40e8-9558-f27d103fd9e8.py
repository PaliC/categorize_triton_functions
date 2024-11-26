import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(X, Y, N):
    idx = tl.program_id(0)
    if idx < N:
        x = tl.load(X + idx)
        y = tl.max(x, 0)
        tl.store(Y + idx, y)
