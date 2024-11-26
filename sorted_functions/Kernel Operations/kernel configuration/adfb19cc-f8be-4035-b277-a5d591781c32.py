import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_INP': 32,
    'BLOCK_SIZE_HIDDEN': 32}, num_stages=3, num_warps=1), triton.Config({
    'BLOCK_SIZE_INP': 64, 'BLOCK_SIZE_HIDDEN': 32}, num_stages=3, num_warps
    =8), triton.Config({'BLOCK_SIZE_INP': 128, 'BLOCK_SIZE_HIDDEN': 32},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_INP': 256,
    'BLOCK_SIZE_HIDDEN': 32}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_INP': 32, 'BLOCK_SIZE_HIDDEN': 64}, num_stages=3, num_warps
    =8), triton.Config({'BLOCK_SIZE_INP': 64, 'BLOCK_SIZE_HIDDEN': 64},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_INP': 128,
    'BLOCK_SIZE_HIDDEN': 64}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_INP': 256, 'BLOCK_SIZE_HIDDEN': 64}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_SIZE_INP': 32, 'BLOCK_SIZE_HIDDEN':
    128}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_INP': 64,
    'BLOCK_SIZE_HIDDEN': 128}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_INP': 128, 'BLOCK_SIZE_HIDDEN': 128}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_SIZE_INP': 256, 'BLOCK_SIZE_HIDDEN':
    128}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_INP': 32,
    'BLOCK_SIZE_HIDDEN': 256}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_INP': 64, 'BLOCK_SIZE_HIDDEN': 256}, num_stages=3,
    num_warps=8), triton.Config({'BLOCK_SIZE_INP': 128, 'BLOCK_SIZE_HIDDEN':
    256}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_INP': 256,
    'BLOCK_SIZE_HIDDEN': 256}, num_stages=3, num_warps=8)], key=['inp_size',
    'hidden_size'], reset_to_zero=['per_slice_sum_ptr'])
@triton.jit
def per_slice_sum_kernel(inp_ptr, upper_end_of_slices_ptr,
    per_slice_sum_ptr, inp_size, hidden_size, num_slices,
    stride_inp_inp_size, stride_inp_hidden_size, BLOCK_SIZE_INP:
    'tl.constexpr', BLOCK_SIZE_HIDDEN: 'tl.constexpr'):
    """Compute the sum per slice."""
    pid = tl.program_id(axis=0)
    offs_am = pid * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)
    ir_range_lower = 0
    for slice_idx in range(0, num_slices):
        ir_range_upper = tl.load(upper_end_of_slices_ptr + slice_idx)
        current_lower = ir_range_lower
        if num_slices == 1:
            num_k = tl.cdiv(ir_range_upper - ir_range_lower, BLOCK_SIZE_HIDDEN)
        else:
            num_k = tl.cdiv(hidden_size, BLOCK_SIZE_HIDDEN)
        offs_k = current_lower + tl.arange(0, BLOCK_SIZE_HIDDEN)
        a_ptrs = inp_ptr + (offs_am[:, None] * stride_inp_inp_size + offs_k
            [None, :] * stride_inp_hidden_size)
        for k in range(0, num_k):
            current_upper = min(ir_range_upper, ir_range_lower + (k + 1) *
                BLOCK_SIZE_HIDDEN, hidden_size)
            inp_block = tl.load(a_ptrs, mask=(offs_am[:, None] < inp_size) &
                (offs_k[None, :] < current_upper), other=0.0)
            inp_block = inp_block
            tl.atomic_add(per_slice_sum_ptr + slice_idx, tl.sum(inp_block))
            offs_k += current_upper - current_lower
            a_ptrs += (current_upper - current_lower) * stride_inp_hidden_size
            current_lower = current_upper
        ir_range_lower = ir_range_upper
