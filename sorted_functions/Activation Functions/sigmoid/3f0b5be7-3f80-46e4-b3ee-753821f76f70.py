import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['D'])
@triton.jit
def logsigmoid_bwd_kernel(x, dx, dy, temperature, T: 'tl.constexpr', D:
    'tl.constexpr', B: 'tl.constexpr'):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T
    b_x = tl.load(x + o_i, mask=m_i, other=0.0)
    b_dy = tl.load(dy + o_i, mask=m_i, other=0.0)
    b_dx = b_dy * (1.0 - tl.sigmoid(b_x)) / temperature
    tl.store(dx + o_i, b_dx, mask=m_i)
