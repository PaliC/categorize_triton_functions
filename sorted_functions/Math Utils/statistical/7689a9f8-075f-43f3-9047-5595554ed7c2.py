import triton
import triton.language as tl
import torch

@triton.jit
def bwd_sequential_scan(grad_output, v, f, h, B, L, C, BLOCK_M: 'tl.constexpr'
    ):
    offset_b = tl.program_id(0)
    if offset_b >= B:
        return
    offset_n = tl.program_id(1)
    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + (L - 1
        ) * C + offset_n * BLOCK_M
    grad_h = tl.zeros([BLOCK_M], dtype=tl.float32)
    for time_step in range(L - 1, -1, -1):
        grad = tl.load(grad_output + ptr)
        grad_h += grad
        decay = tl.load(f + ptr)
        input = tl.load(v + ptr)
        grad_v = (1 - decay) * grad_h
        tl.store(v + ptr, grad_v)
        hidden_state = tl.load(h + ptr - C, mask=ptr >= offset_b * L * C +
            C, other=0.0)
        grad_f = grad_h * (hidden_state - input)
        tl.store(f + ptr, grad_f)
        grad_h *= decay
        ptr -= C
