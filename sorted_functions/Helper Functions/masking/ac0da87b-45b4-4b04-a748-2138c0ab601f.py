import triton
import triton.language as tl
import torch

@triton.heuristics({'HAS_SMOOTHING': lambda args: args['smoothing'] > 0.0})
@triton.jit
def cross_entropy_bwd_kernel(dlogits_ptr, dloss_ptr, logits_ptr, lse_ptr,
    labels_ptr, smoothing, lse_square_scale, ignored_index, total_classes,
    class_start_idx, n_cols, logits_row_stride, dlogits_row_stride,
    dloss_row_stride, BLOCK_SIZE: 'tl.constexpr', HAS_SMOOTHING: 'tl.constexpr'
    ):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignored_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols,
        other=-float('inf'))
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_positive = 1.0 - smoothing
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing),
            probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, dloss * probs, mask=col_offsets <
        n_cols)
