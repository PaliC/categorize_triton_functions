import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4, 8, 16, 32]], key=['num_batches', 'num_frames',
    'fft_size'])
@triton.jit
def complex_mul_conjugate_kernel(a_real_ptr, b_real_ptr, a_imag_ptr,
    b_imag_ptr, output1_ptr, output2_ptr, num_batches, num_frames, fft_size,
    BLOCK_SIZE: 'tl.constexpr'):
    batch_idx = tl.program_id(0)
    if batch_idx >= num_batches:
        return
    fft_idx = tl.arange(0, BLOCK_SIZE)
    fft_mask = fft_idx < fft_size
    batch_by_fft = batch_idx * fft_size
    b_real_val = tl.load(b_real_ptr + batch_by_fft + fft_idx, mask=fft_mask)
    b_imag_val = tl.load(b_imag_ptr + batch_by_fft + fft_idx, mask=fft_mask)
    for frame_idx in range(num_frames):
        global_idx = num_frames * batch_by_fft + frame_idx * fft_size + fft_idx
        a_real_val = tl.load(a_real_ptr + global_idx, mask=fft_mask)
        a_imag_val = tl.load(a_imag_ptr + global_idx, mask=fft_mask)
        result1 = a_real_val * b_real_val + a_imag_val * b_imag_val
        result2 = a_imag_val * b_real_val - a_real_val * b_imag_val
        tl.store(output1_ptr + global_idx, result1, mask=fft_mask)
        tl.store(output2_ptr + global_idx, result2, mask=fft_mask)
