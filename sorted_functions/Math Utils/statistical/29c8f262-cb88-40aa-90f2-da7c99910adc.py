import triton
import triton.language as tl
import torch

@triton.jit
def fused_cross_entropy_fwd_bwd_kernel(output_loss_ptr,
    output_logit_grad_ptr, input_logit_ptr, input_targ_ptr,
    input_divisor_ptr, output_loss_stride, output_logit_grad_stride,
    input_logit_stride, input_targ_stride, n_cols, ignore_index, BLOCK_SIZE:
    'tl.constexpr'):
    row_idx = tl.program_id(0)
    logit_grad_row_start_ptr = (output_logit_grad_ptr + row_idx *
        output_logit_grad_stride)
    logit_row_start_ptr = input_logit_ptr + row_idx * input_logit_stride
    targ_ptr = input_targ_ptr + row_idx * input_targ_stride
    loss_ptr = output_loss_ptr + row_idx * output_loss_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    logit_row_ptrs = logit_row_start_ptr + col_offsets
    logit_grad_row_ptrs = logit_grad_row_start_ptr + col_offsets
    logit_row_unnormalized = tl.load(logit_row_ptrs, mask=col_offsets <
        n_cols, other=float('-Inf'))
    targ = tl.load(targ_ptr)
    divisor = tl.load(input_divisor_ptr)
    logit_row = logit_row_unnormalized - tl.max(logit_row_unnormalized, axis=0)
    exp_logit_row = tl.exp(logit_row)
    sum_exp_logit_row = tl.sum(exp_logit_row, axis=0)
    log_sum_exp_logit_row = tl.log(sum_exp_logit_row)
    logit_gt_logit = tl.sum(tl.where(targ == col_offsets, logit_row, 0.0))
    loss = log_sum_exp_logit_row - logit_gt_logit
    loss = loss / divisor
    loss = tl.where(targ == ignore_index, 0.0, loss)
    tl.store(loss_ptr, loss)
    targ_one_hot = tl.where(targ == col_offsets, 1.0, 0.0)
    grad = exp_logit_row / sum_exp_logit_row - targ_one_hot
    grad = grad / divisor
    grad = tl.where(targ == ignore_index, 0.0, grad)
    tl.store(logit_grad_row_ptrs, grad, mask=col_offsets < n_cols)
