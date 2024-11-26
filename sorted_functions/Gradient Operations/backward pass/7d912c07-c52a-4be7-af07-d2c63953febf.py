import triton
import triton.language as tl
import torch

@triton.jit
def bwd_sequential_scan_complex(grad_output_real, grad_output_imag, v_real,
    v_imag, f_real, f_imag, hidden_real, hidden_imag, grad_detach, B, L, C,
    BLOCK_M: 'tl.constexpr'):
    offset_b = tl.program_id(0)
    if offset_b >= B:
        return
    offset_n = tl.program_id(1)
    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + (L - 1
        ) * C + offset_n * BLOCK_M
    grad_detach_ptr = grad_detach + offset_b * L + (L - 1)
    grad_h_real = tl.zeros([BLOCK_M], dtype=tl.float32)
    grad_h_imag = tl.zeros([BLOCK_M], dtype=tl.float32)
    for time_step in range(L - 1, -1, -1):
        grad_real = tl.load(grad_output_real + ptr)
        grad_imag = tl.load(grad_output_imag + ptr)
        grad_detach_item = tl.load(grad_detach_ptr)
        grad_h_real = grad_h_real * (1 - grad_detach_item)
        grad_h_imag = grad_h_imag * (1 - grad_detach_item)
        grad_h_real += grad_real
        grad_h_imag += grad_imag
        decay_real = tl.load(f_real + ptr)
        decay_imag = tl.load(f_imag + ptr)
        h_real = tl.load(hidden_real + ptr)
        h_imag = tl.load(hidden_imag + ptr)
        grad_f_real = grad_h_real * h_real + grad_h_imag * h_imag
        grad_f_imag = grad_h_imag * h_real - grad_h_real * h_imag
        tl.store(f_real + ptr, grad_f_real)
        tl.store(f_imag + ptr, grad_f_imag)
        tl.store(v_real + ptr, grad_h_real)
        tl.store(v_imag + ptr, grad_h_imag)
        grad_h_real_new = grad_h_real * decay_real + grad_h_imag * decay_imag
        grad_h_imag_new = grad_h_imag * decay_real - grad_h_real * decay_imag
        grad_h_real = grad_h_real_new
        grad_h_imag = grad_h_imag_new
        ptr -= C
        grad_detach_ptr -= 1
