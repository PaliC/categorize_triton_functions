import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['D'])
@triton.jit
def softmax_bwd_kernel(p, dp, ds, D: 'tl.constexpr', B: 'tl.constexpr'):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < D
    b_p = tl.load(p + i_n * D + o_d, mask=m_d, other=0.0)
    b_dp = tl.load(dp + i_n * D + o_d, mask=m_d, other=0.0)
    b_pp = tl.sum(b_p * b_dp, 0)
    b_ds = b_p * b_dp - b_p * b_pp
    tl.store(ds + i_n * D + o_d, b_ds, mask=m_d)
