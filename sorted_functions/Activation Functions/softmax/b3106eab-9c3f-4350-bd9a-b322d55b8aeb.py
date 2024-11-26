import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['D'])
@triton.jit
def softmax_fwd_kernel(x, p, D: 'tl.constexpr', B: 'tl.constexpr'):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < D
    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    b_m = tl.max(b_x, 0)
    b_x = tl.exp(b_x - b_m)
    b_p = b_x / tl.sum(b_x, 0)
    tl.store(p + i_n * D + o_d, b_p, mask=m_d)
