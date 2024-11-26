import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_loss_kernel(logits_ptr, targets_ptr, loss_ptr, n_classes,
    n_elements, BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    targets = tl.load(targets_ptr + offsets, mask=mask, other=-1)
    row_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(n_classes):
        col_offset = offsets * n_classes + i
        logit = tl.load(logits_ptr + col_offset, mask=mask, other=float('-inf')
            )
        row_max = tl.maximum(row_max, logit)
    loss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(n_classes):
        col_offset = offsets * n_classes + i
        logit = tl.load(logits_ptr + col_offset, mask=mask, other=float('-inf')
            )
        exp_logit = tl.exp(logit - row_max)
        row_sum += exp_logit
        loss = tl.where(targets == i, loss - logit + row_max, loss)
    loss += tl.log(row_sum)
    tl.store(loss_ptr + offsets, loss, mask=mask)
