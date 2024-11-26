import triton
import triton.language as tl
import torch

@triton.jit
def __triton_round_compute(X, stride_x_n, N, BLOCK_N: 'tl.constexpr'):
    pid_n = tl.program_id(0)
    grid_n = tl.num_programs(0)
    n = tl.arange(0, BLOCK_N) * grid_n + pid_n
    n_mask = n < N
    xs = tl.load(X + n * stride_x_n, mask=n_mask)
    ys = tl.math.round(xs)
    tl.store(X + n * stride_x_n, ys, mask=n_mask)
