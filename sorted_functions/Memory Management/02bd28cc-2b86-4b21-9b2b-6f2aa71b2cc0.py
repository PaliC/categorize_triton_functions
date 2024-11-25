import triton
import triton.language as tl
import torch

@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location,
    new_a_start, new_a_location, BLOCK_SIZE: 'tl.constexpr'):
    a_id = tl.program_id(0)
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)
    old_offset = tl.arange(0, BLOCK_SIZE)
    new_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + old_start + i + old_offset, mask=
            old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=
            new_offset < length)
