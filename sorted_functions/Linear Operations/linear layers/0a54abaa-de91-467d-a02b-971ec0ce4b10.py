import triton
import triton.language as tl
import torch

@triton.jit
def fwd_sequential_scan_fused(v, f1, hidden, B, L, C, BLOCK_M: 'tl.constexpr'):
    offset_b = tl.program_id(0)
    if offset_b >= B:
        return
    offset_n = tl.program_id(1)
    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + offset_n * BLOCK_M
    h1 = tl.zeros([BLOCK_M], dtype=tl.float32)
    for _ in range(L):
        x0 = tl.load(v + ptr)
        decay1 = tl.load(f1 + ptr)
        decay1 = tl.sigmoid(decay1)
        h1 = (h1 - x0) * decay1 + x0
        tl.store(hidden + ptr, h1)
        ptr += C
