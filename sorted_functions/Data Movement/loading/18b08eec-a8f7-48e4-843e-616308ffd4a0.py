import triton
import triton.language as tl
import torch

@triton.jit
def mload1d(REGS: 'tl.constexpr', i_base, i_start, i_nums):
    offs = tl.arange(0, REGS) + i_start
    i_ptrs = i_base + offs
    overflow = i_start + REGS - i_nums
    i_ptrs_mask = tl.full([REGS], 1, dtype=tl.int1)
    i_ptrs_mask = i_ptrs_mask & (offs < i_nums)
    return tl.load(i_ptrs, mask=i_ptrs_mask, other=0.0)
