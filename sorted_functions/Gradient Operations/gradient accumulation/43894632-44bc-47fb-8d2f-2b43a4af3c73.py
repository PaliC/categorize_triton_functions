import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_X': 64}, num_warps=2),
    triton.Config({'BLOCK_X': 128}, num_warps=2), triton.Config({'BLOCK_X':
    256}, num_warps=2), triton.Config({'BLOCK_X': 128}, num_warps=4),
    triton.Config({'BLOCK_X': 256}, num_warps=4)], key=['NUM_COLUMNS'])
@triton.jit
def _padded_copy_wgrad(x, grad, wgrad, indices, bin_ids, bins, padded_bins,
    NUM_COLUMNS: 'tl.constexpr', TOP_K: 'tl.constexpr', BLOCK_X: 'tl.constexpr'
    ):
    index_out = tl.load(indices + tl.program_id(0))
    bin_idx = tl.load(bin_ids + tl.program_id(0))
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)
    index_x = offset_in_bin
    if bin_idx > 0:
        index_x += tl.load(padded_bins + bin_idx - 1)
    wgrad += index_out
    grad += tl.multiple_of(index_out // TOP_K * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)
    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for i in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask)
        scale = tl.load(grad + offsets, mask=mask)
        acc += data * scale
        offsets += BLOCK_X
    out = tl.sum(acc)
    tl.store(wgrad, out)
