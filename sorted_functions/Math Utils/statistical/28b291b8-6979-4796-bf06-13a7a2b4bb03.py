import triton
import triton.language as tl
import torch

@triton.jit
def coor_descent_kernel_forward(a_ptr, b_ptr, input_ptr, mask_ptr, k_ptr,
    a_iter_stride, b_row_stride, b_iter_stride, input_row_stride,
    mask_row_stride, n_iters, current_eps, eps_decay, eps, n_cols,
    BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets
    mask_ints = tl.load(mask_ptrs, mask=col_mask, other=0)
    mask = mask_ints == 1
    a_ptr = a_ptr + row_idx
    a = tl.load(a_ptr)
    b_start_ptr = b_ptr + row_idx * b_row_stride
    b_ptrs = b_start_ptr + col_offsets
    b = tl.load(b_ptrs, mask=col_mask, other=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    s = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    k_ptr = k_ptr + row_idx
    k = tl.load(k_ptr)
    logk = tl.log(k)
    for _ in range(n_iters):
        a = (s + b) / current_eps
        a = tl.where(mask, a, -float('inf'))
        a_max = tl.max(a, axis=0)
        a_minus_max = tl.where(mask, a - a_max, -float('inf'))
        exp = tl.exp(a_minus_max)
        sum_exp = tl.sum(exp, axis=0)
        log_sum_exp = tl.log(sum_exp) + a_max
        a = current_eps * (logk - log_sum_exp)
        b = s + a
        b = tl.where(b >= 0.0, -b, 0.0)
        current_eps *= eps_decay
        if current_eps < eps:
            current_eps = eps
    next_a_ptrs = a_ptr + a_iter_stride
    next_b_ptrs = b_ptrs + b_iter_stride
    tl.store(next_a_ptrs, a)
    tl.store(next_b_ptrs, b, mask=col_mask)
