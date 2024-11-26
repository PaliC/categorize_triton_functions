import triton
import triton.language as tl
import torch

@triton.jit
def fwd_sequential_scan_complex(v_real, v_imag, decay_real, decay_imag,
    hidden_real, hidden_imag, hidden_real_input, hidden_imag_input, B, L, C,
    BLOCK_M: 'tl.constexpr'):
    offset_b = tl.program_id(0)
    if offset_b >= B:
        return
    offset_n = tl.program_id(1)
    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + offset_n * BLOCK_M
    ptr_input_hidden = tl.arange(0, BLOCK_M
        ) + offset_b * C + offset_n * BLOCK_M
    h_real = tl.load(hidden_real_input + ptr_input_hidden)
    h_imag = tl.load(hidden_imag_input + ptr_input_hidden)
    for _ in range(L):
        x_real = tl.load(v_real + ptr)
        x_imag = tl.load(v_imag + ptr)
        f_real = tl.load(decay_real + ptr)
        f_imag = tl.load(decay_imag + ptr)
        h_real_new = h_real * f_real - h_imag * f_imag + x_real
        h_imag_new = h_real * f_imag + h_imag * f_real + x_imag
        tl.store(hidden_real + ptr, h_real_new)
        tl.store(hidden_imag + ptr, h_imag_new)
        h_real = h_real_new
        h_imag = h_imag_new
        ptr += C
