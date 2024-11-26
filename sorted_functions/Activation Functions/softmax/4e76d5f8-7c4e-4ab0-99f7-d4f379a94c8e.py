import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel_backward(grad_out_ptr, probs_ptr, grad_in_ptr,
    grad_stride, probs_stride, out_stride, seq_len, BLOCK_SIZE:
    'tl.constexpr', num_warps: 'tl.constexpr'):
    batch_idx = tl.program_id(0)
    probs_start_ptr = probs_ptr + batch_idx * probs_stride
    grad_start_ptr = grad_in_ptr + batch_idx * grad_stride
    pos_offsets = tl.arange(0, BLOCK_SIZE)
    probs_ptrs = probs_start_ptr + pos_offsets
    grad_ptrs = grad_start_ptr + pos_offsets
    valid_mask = pos_offsets < seq_len
    probs_vals = tl.load(probs_ptrs, mask=valid_mask, other=0.0)
    grad_vals = tl.load(grad_ptrs, mask=valid_mask, other=0.0)
    grad_times_probs = probs_vals * grad_vals
    final_grad = grad_times_probs - probs_vals * tl.sum(grad_times_probs,
        axis=0)
    out_start_ptr = grad_out_ptr + batch_idx * out_stride
    out_ptrs = out_start_ptr + pos_offsets
    tl.store(out_ptrs, final_grad, mask=valid_mask)
