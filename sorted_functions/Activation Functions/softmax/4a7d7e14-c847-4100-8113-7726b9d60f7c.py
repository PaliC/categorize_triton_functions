import triton
import triton.language as tl
import torch

@triton.jit
def triton_softmax(X_ptr, Y_ptr, M, N, BLOCK_SIZE):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < M
    x_row = tl.load(X_ptr + idx * N, mask=mask)
    x_max = tl.max(x_row)
    x_shifted = x_row - x_max
    exp_x = tl.exp(x_shifted)
    sum_x = tl.sum(exp_x)
    softmax_ret = exp_x / sum_x
    tl.store(Y_ptr + idx * N, softmax_ret, mask=mask)
