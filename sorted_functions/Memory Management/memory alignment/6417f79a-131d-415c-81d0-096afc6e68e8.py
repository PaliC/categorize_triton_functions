import triton
import triton.language as tl
import torch

@triton.jit
def tl_lock_add(ptrs, v, mask, lock_ptr):
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass
    cur_v = tl.load(ptrs, mask=mask, other=0.0, eviction_policy='evict_last')
    new_v = v + cur_v
    tl.store(ptrs, new_v, mask=mask, eviction_policy='evict_last')
    tl.atomic_xchg(lock_ptr, 0)
