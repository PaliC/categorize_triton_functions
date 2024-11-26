import triton
import triton.language as tl
import torch

@triton.jit
def coor_descent_kernel_backward(dk_ptr, input_ptr, a_ptr, b_ptr, mask_ptr,
    ds_ptr, db_ptr, k_ptr, last_da_ptr, input_row_stride, b_row_stride,
    mask_row_stride, ds_row_stride, db_row_stride, n_iters, eps_init,
    eps_decay, eps, n_cols, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets
    mask_ints = tl.load(mask_ptrs, mask=col_mask, other=0)
    mask = mask_ints == 1
    a_ptr = a_ptr + row_idx
    init_a = tl.load(a_ptr)
    b_start_ptr = b_ptr + row_idx * b_row_stride
    b_ptrs = b_start_ptr + col_offsets
    init_b = tl.load(b_ptrs, mask=mask, other=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    s = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    k_ptr = k_ptr + row_idx
    k = tl.load(k_ptr)
    logk = tl.log(k)
    last_da_ptr = last_da_ptr + row_idx
    last_da = tl.load(last_da_ptr)
    ds_row_start_ptr = ds_ptr + row_idx * ds_row_stride
    ds_ptrs = ds_row_start_ptr + col_offsets
    ds = tl.load(ds_ptrs, mask=mask, other=0.0)
    db_row_start_ptr = db_ptr + row_idx * db_row_stride
    db_ptrs = db_row_start_ptr + col_offsets
    db = tl.load(db_ptrs, mask=mask, other=0.0)
    dk_ptr = dk_ptr + row_idx
    dk = tl.load(dk_ptr)
    for ind in range(n_iters):
        a = init_a
        b = init_b
        sa = s * 0
        softmax = s * 0
        current_eps = eps_init / eps_decay
        for _ in range(n_iters - ind):
            current_eps *= eps_decay
            if current_eps < eps:
                current_eps = eps
            sb = (s + b) / current_eps
            sb = tl.where(mask, sb, -float('inf'))
            sb_max = tl.max(sb, axis=0)
            sb_minus_max = tl.where(mask, sb - sb_max, -float('inf'))
            exp = tl.exp(sb_minus_max)
            sum_exp = tl.sum(exp, axis=0)
            softmax = exp / sum_exp
            log_sum_exp = tl.log(sum_exp) + sb_max
            a = current_eps * (logk - log_sum_exp)
            sa = s + a
            b = tl.where(sa > 0.0, -sa, 0.0)
        dsa = db * tl.where(sa > 0, -1.0, 0.0)
        ds += dsa
        da = tl.sum(dsa, axis=0) + last_da
        dk += da * current_eps
        dsb = da * -softmax
        ds += dsb
        db = dsb
        last_da *= 0.0
    tl.store(dk_ptr, dk)
    tl.store(ds_ptrs, ds, mask=col_mask)
    tl.store(db_ptrs, db, mask=col_mask)
