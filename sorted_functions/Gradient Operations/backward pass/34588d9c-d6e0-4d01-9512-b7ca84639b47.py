import triton
import triton.language as tl
import torch

@triton.jit
def _mm_backward(do, da_ptrs, partial_mask_a, da_lock_ptr, n_locks, b_ptrs,
    partial_mask_b, stride_ad, stride_bd, D, BLOCK_D: 'tl.constexpr',
    EVEN_D: 'tl.constexpr'):
    d_inds = tl.arange(0, BLOCK_D)[None, :]
    da_ptrs = da_ptrs + d_inds * stride_ad
    b_ptrs = b_ptrs + d_inds * stride_bd
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            mask = partial_mask_b
        else:
            mask = partial_mask_b & (d_inds < D - d * BLOCK_D)
        b = tl.load(b_ptrs, mask=mask, other=0.0)
        da_i = tl.dot(do, b)
        if EVEN_D:
            mask = partial_mask_a
        else:
            mask = partial_mask_a & (d_inds < D - d * BLOCK_D)
        lock_offset = d // tl.cdiv(D, BLOCK_D * n_locks)
        this_da_lock_ptr = da_lock_ptr + lock_offset
        tl_lock_add(da_ptrs, da_i, mask, this_da_lock_ptr)
        b_ptrs += BLOCK_D * stride_bd
        da_ptrs += BLOCK_D * stride_ad
