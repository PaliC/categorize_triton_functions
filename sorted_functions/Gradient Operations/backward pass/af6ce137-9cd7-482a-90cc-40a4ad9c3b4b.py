import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_preprocess(Out, DO, Delta, BLOCK_M: 'tl.constexpr', D_HEAD:
    'tl.constexpr'):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :])
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :])
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_m, delta)
