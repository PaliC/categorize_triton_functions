import triton
import triton.language as tl
import torch

@triton.jit
def _complex_operator_element_(_x_real_a, _a_imag_a, _x_imag_a, _a_real_a,
    start, num, interval, offset_b, offset_n, L, C, last_interval, BLOCK_M:
    'tl.constexpr'):
    offset_t = tl.program_id(0)
    range_batch = tl.arange(0, BLOCK_M) + offset_b * L * C + offset_n * BLOCK_M
    range_time = (tl.arange(0, num) * interval + start) * C
    range_2dim = range_batch[:, None] + range_time[None, :]
    ptr = range_2dim
    ptr_last = range_2dim - last_interval * C
    x_real_a = tl.load(_x_real_a + ptr)
    x_real_a_last = tl.load(_x_real_a + ptr_last)
    a_imag_a = tl.load(_a_imag_a + ptr)
    a_imag_a_last = tl.load(_a_imag_a + ptr_last)
    x_imag_a = tl.load(_x_imag_a + ptr)
    x_imag_a_last = tl.load(_x_imag_a + ptr_last)
    a_real_a = tl.load(_a_real_a + ptr)
    a_real_a_last = tl.load(_a_real_a + ptr_last)
    x_real_a = x_real_a + a_real_a * x_real_a_last - a_imag_a * x_imag_a_last
    x_imag_a = x_imag_a + a_real_a * x_imag_a_last + a_imag_a * x_real_a_last
    tl.store(_x_real_a + ptr, x_real_a)
    tl.store(_x_imag_a + ptr, x_imag_a)
    a_real_a_next = a_real_a * a_real_a_last - a_imag_a * a_imag_a_last
    a_imag_a_next = a_imag_a * a_real_a_last - a_real_a * a_imag_a_last
    tl.store(_a_real_a + ptr, a_real_a_next)
    tl.store(_a_imag_a + ptr, a_imag_a_next)
