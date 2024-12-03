import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_X': 64}, num_warps=2),
    triton.Config({'BLOCK_X': 128}, num_warps=2), triton.Config({'BLOCK_X':
    256}, num_warps=2), triton.Config({'BLOCK_X': 128}, num_warps=4),
    triton.Config({'BLOCK_X': 256}, num_warps=4)], key=['NUM_COLUMNS'])
@triton.jit
def _padded_copy(a, b, indices, bin_ids, weights, bins, padded_bins,
    NUM_COLUMNS: 'tl.constexpr', TOP_K: 'tl.constexpr', BLOCK_X:
    'tl.constexpr', A_TO_B: 'tl.constexpr', SCALE: 'tl.constexpr'):
    index_a = tl.load(indices + tl.program_id(0))
    bin_idx = tl.load(bin_ids + tl.program_id(0))
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)
    index_b = offset_in_bin
    if bin_idx > 0:
        index_b += tl.load(padded_bins + bin_idx - 1)
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)
    scale = tl.load(weights + index_a) if SCALE else 1
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a
    for i in range(tl.cdiv(NUM_COLUMNS, BLOCK_X)):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x * scale
        tl.store(optr + offsets, x, mask=mask)
        offsets += BLOCK_X
