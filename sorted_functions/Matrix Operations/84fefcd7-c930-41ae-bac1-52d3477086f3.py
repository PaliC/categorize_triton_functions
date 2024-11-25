import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['D'])
@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None})
@triton.jit
def logsumexp_fwd_kernel(x, z, scale, D: 'tl.constexpr', B: 'tl.constexpr',
    HAS_SCALE: 'tl.constexpr'):
    i_n, i_d = tl.program_id(0), tl.program_id(1)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D
    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
    b_m = tl.max(b_x, 0)
    b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)
