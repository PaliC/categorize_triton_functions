import triton
import triton.language as tl
import torch

@triton.jit
def grouped_launch(pid, m, n, block_m: 'tl.constexpr', block_n:
    'tl.constexpr', group_m: 'tl.constexpr'):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)
    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + pid % group_size
    pid_n = pid % width // group_size
    return pid_m, pid_n
