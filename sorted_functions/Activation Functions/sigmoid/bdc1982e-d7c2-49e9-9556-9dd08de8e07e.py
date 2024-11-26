import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['D'])
@triton.jit
def logsigmoid_fwd_kernel(x, y, temperature, T: 'tl.constexpr', D:
    'tl.constexpr', B: 'tl.constexpr'):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T
    b_x = tl.load(x + o_i, mask=m_i, other=0.0)
    b_m = tl.minimum(0.0, b_x)
    b_z = 1.0 + tl.exp(-tl.abs(b_x))
    b_y = (b_m - tl.log(b_z)) / temperature
    tl.store(y + o_i, b_y, mask=m_i)
