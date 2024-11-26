import triton
import triton.language as tl
import torch

@triton.jit
def _row_indices_kernel(offsets, out):
    pid = tl.program_id(0)
    row_offset = tl.load(offsets + pid)
    nnz_blocks = tl.load(offsets + pid + 1) - row_offset
    for nnz_block in range(nnz_blocks):
        tl.store(out + row_offset + nnz_block, pid)
