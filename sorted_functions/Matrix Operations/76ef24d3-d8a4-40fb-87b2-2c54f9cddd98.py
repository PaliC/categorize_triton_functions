import triton
import triton.language as tl
import torch

@triton.jit
def kernel_vector_addition(a_ptr, b_ptr, out_ptr, num_elems: 'tl.constexpr',
    block_size: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    thread_offsets = block_start + tl.arange(0, block_size)
    mask = thread_offsets < num_elems
    a_pointers = tl.load(a_ptr + thread_offsets, mask=mask)
    b_pointers = tl.load(b_ptr + thread_offsets, mask=mask)
    res = a_pointers + b_pointers
    tl.store(out_ptr + thread_offsets, res, mask=mask)
