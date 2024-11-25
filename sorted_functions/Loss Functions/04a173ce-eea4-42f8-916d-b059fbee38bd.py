import triton
import triton.language as tl
import torch

@triton.heuristics({'HAS_SMOOTHING': lambda args: args['label_smoothing'] >
    0.0})
@triton.jit
def cross_entropy_fwd_kernel(loss_ptr, lse_ptr, z_loss_ptr, logits_ptr,
    labels_ptr, label_smoothing, logit_scale, lse_square_scale,
    ignore_index, total_classes, class_start_idx, n_cols, n_rows,
    logits_row_stride, BLOCK_SIZE: 'tl.constexpr', HAS_SMOOTHING:
    'tl.constexpr', SPLIT: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols,
        other=-float('inf'))
    logits = logits * logit_scale
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(n_cols,
            (col_block_idx + 1) * BLOCK_SIZE):
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (lse if not SPLIT else 0.0
                    ) - label_smoothing * sum_logits / total_classes - (1 -
                    label_smoothing) * logits_label
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        elif HAS_SMOOTHING:
            loss = label_smoothing * ((lse if not SPLIT else 0.0) - 
                sum_logits / total_classes)
        else:
            loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)
