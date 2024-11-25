import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.
    Config({'BT': 16}, num_warps=4), triton.Config({'BT': 16}, num_warps=8),
    triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 32},
    num_warps=4), triton.Config({'BT': 32}, num_warps=8), triton.Config({
    'BT': 64}, num_warps=2), triton.Config({'BT': 64}, num_warps=4), triton
    .Config({'BT': 64}, num_warps=8), triton.Config({'BT': 128}, num_warps=
    2), triton.Config({'BT': 128}, num_warps=4), triton.Config({'BT': 128},
    num_warps=8), triton.Config({'BT': 256}, num_warps=2), triton.Config({
    'BT': 256}, num_warps=4), triton.Config({'BT': 256}, num_warps=8)], key
    =['D'])
@triton.jit
def logsigmoid_fwd_kernel(x, y, T: 'tl.constexpr', D: 'tl.constexpr', BT:
    'tl.constexpr'):
    i = tl.program_id(0)
    o_i = i * BT + tl.arange(0, BT)
    p_x = x + o_i
    p_y = y + o_i
    mask = o_i < T
    b_x = tl.load(p_x, mask=mask, other=0.0)
    b_m = tl.minimum(0.0, b_x)
    b_z = 1.0 + tl.exp(-tl.abs(b_x))
    b_y = b_m - tl.log(b_z)
    tl.store(p_y, b_y, mask=mask)
