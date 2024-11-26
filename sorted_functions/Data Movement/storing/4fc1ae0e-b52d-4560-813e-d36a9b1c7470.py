import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_store_dx(dx_ptrs, dx, offs_d, headdim, even_headdim):
    if even_headdim:
        tl.store(dx_ptrs, dx)
    else:
        tl.store(dx_ptrs, dx, mask=offs_d[None, :] < headdim)
