import triton
import triton.language as tl
import torch

@triton.jit
def add_func(a_ptr, b_ptr, c_ptr, element_size, block_size: 'tl.constexpr'):
    """Triton Func."""
    block_id = tl.program_id(axis=0)
    thread_id = block_id * block_size + tl.arange(0, block_size)
    mask = thread_id < element_size
    a = tl.load(a_ptr + thread_id, mask=mask)
    b = tl.load(b_ptr + thread_id, mask=mask)
    tl.store(c_ptr + thread_id, a + b, mask=mask)
