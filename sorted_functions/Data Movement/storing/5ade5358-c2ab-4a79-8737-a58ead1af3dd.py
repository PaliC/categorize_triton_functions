import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_store_dx(dx_ptrs, dx, offs_n, offs_d, seqlen, headdim, EVEN_M:
    'tl.constexpr', EVEN_N: 'tl.constexpr', even_headdim):
    if EVEN_N & EVEN_M:
        if even_headdim:
            tl.store(dx_ptrs, dx)
        else:
            tl.store(dx_ptrs, dx, mask=offs_d[None, :] < headdim)
    elif even_headdim:
        tl.store(dx_ptrs, dx, mask=offs_n[:, None] < seqlen)
    else:
        tl.store(dx_ptrs, dx, mask=(offs_n[:, None] < seqlen) & (offs_d[
            None, :] < headdim))
