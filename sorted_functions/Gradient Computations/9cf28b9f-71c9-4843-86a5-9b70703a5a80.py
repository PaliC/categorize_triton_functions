import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_preprocess(Out, soz, soh, som, sod, DO, L, slzh, slm, NewDO, Delta,
    N_CTX_Q, BLOCK_M: 'tl.constexpr', D_HEAD: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_HEAD)
    off_o = off_hz * soh + off_m[:, None] * som + off_d[None, :] * sod
    off_l = off_hz * slzh + off_m * slm
    o = tl.load(Out + off_o)
    do = tl.load(DO + off_o)
    denom = tl.load(L + off_l)
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    tl.store(NewDO + off_o, do)
    tl.store(Delta + off_l, delta)
