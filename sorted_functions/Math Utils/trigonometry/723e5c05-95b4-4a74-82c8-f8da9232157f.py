import triton
import triton.language as tl
import torch

@triton.jit
def sinc_kernel(output_ptr, cutoffs_ptr, indices_ptr, num_taps, window_ptr,
    half_sample_rate, mode: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    batch_idx = tl.program_id(1)
    pos = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pos < num_taps
    cutoff_val = tl.load(cutoffs_ptr + batch_idx) / half_sample_rate
    index_val = tl.load(indices_ptr + pos, mask=mask)
    window_val = tl.load(window_ptr + pos, mask=mask)
    x = index_val * math.pi * cutoff_val
    sinc_val = tl.where(index_val == 0, 1.0, tl.sin(x) / x)
    windowed_sinc = sinc_val * window_val
    normalized_sinc = windowed_sinc / tl.sum(windowed_sinc, axis=0)
    if mode == 'high':
        center_idx = num_taps // 2
        adjusted_val = tl.where(pos == center_idx, 1.0 - normalized_sinc, -
            normalized_sinc)
        tl.store(output_ptr + batch_idx * num_taps + pos, adjusted_val,
            mask=mask)
    elif mode == 'low':
        tl.store(output_ptr + batch_idx * num_taps + pos, normalized_sinc,
            mask=mask)
    else:
        raise ValueError(f'Unknown mode: {mode}')
