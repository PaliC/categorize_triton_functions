import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel_forward(out_ptr, inp_ptr, inp_stride, out_stride,
    seq_len, is_causal, BLOCK_SIZE: 'tl.constexpr', num_warps: 'tl.constexpr'):
    batch_idx = tl.program_id(0)
    batch_start_ptr = inp_ptr + batch_idx * inp_stride
    pos_offsets = tl.arange(0, BLOCK_SIZE)
    batch_ptrs = batch_start_ptr + pos_offsets
    valid_mask = pos_offsets < seq_len
    logits = tl.load(batch_ptrs, mask=valid_mask, other=-float('inf'))
    if is_causal:
        attn_mask = pos_offsets > batch_idx % seq_len
        logits = logits + tl.where(attn_mask, -float('inf'), 0.0)
    shifted_logits = logits - tl.max(logits, axis=0)
    exp_logits = tl.exp(shifted_logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp
    out_batch_ptr = out_ptr + batch_idx * out_stride
    out_ptrs = out_batch_ptr + pos_offsets
    tl.store(out_ptrs, probs, mask=valid_mask)
