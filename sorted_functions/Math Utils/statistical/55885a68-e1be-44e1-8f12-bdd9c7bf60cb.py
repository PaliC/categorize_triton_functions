import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_SUM': block_size_sum},
    num_warps=num_warps) for block_size_sum, num_warps in itertools.product
    ([512, 1024], [2, 4, 8, 16])], key=['clean_audio_max_len',
    'noisy_audio_max_len'])
@triton.jit
def sum_with_snr_kernel(clean_audio, clean_audio_real_lens,
    clean_audio_max_len, desired_rms, noisy_audio_ptr,
    noisy_audio_real_lens, noisy_audio_max_len, output_ptr, BLOCK_SIZE_SUM:
    'tl.constexpr', BLOCK_SIZE_RMS: 'tl.constexpr'):
    batch_idx = tl.program_id(0)
    clean_audio_real_lens_val = tl.load(clean_audio_real_lens + batch_idx)
    clean_audio_rms = rms_kernel(clean_audio, clean_audio_real_lens,
        clean_audio_max_len, batch_idx, BLOCK_SIZE_RMS)
    noisy_audio_real_lens_val = tl.load(noisy_audio_real_lens + batch_idx)
    noisy_audio_rms = rms_kernel(noisy_audio_ptr, noisy_audio_real_lens,
        noisy_audio_max_len, batch_idx, BLOCK_SIZE_RMS)
    desired_rms_val = tl.load(desired_rms + batch_idx)
    relative_rms = clean_audio_rms / tl.math.pow(10.0, desired_rms_val / 20.0)
    for offset in range(0, clean_audio_max_len, BLOCK_SIZE_SUM):
        clean_audio_block_ptr = offset + tl.arange(0, BLOCK_SIZE_SUM)
        clean_audio_mask = clean_audio_block_ptr < clean_audio_real_lens_val
        clean_audio_vals = tl.load(clean_audio + batch_idx *
            clean_audio_max_len + clean_audio_block_ptr, mask=clean_audio_mask)
        """
        Adjusts the block's start position if it extends beyond the noisy audio array, shifting it leftward as needed.
        This adjustment keeps the data block within the noisy audio array limits, accounting for its circular nature

        Scenario without adjustment:
           noisy_audio_array: |----|----|----|----|----|----|----|----|
           block:                      |~~~~~~~~~~~~~~~~|
           (Block fits within the array, no adjustment needed)

        Scenario with adjustment:
           noisy_audio_array: |----|----|----|----|----|----|----|----|
           block:                                           |~~~~~~~~~~~~~~~~|
           (Block exceeds array bounds, needs to be shifted left)
           noisy_audio_array: |----|----|----|----|----|----|----|----|
           block:                                    |~~~~~~~~~~~~~~~~|  <--- Shifted left
        """
        offset_over_max = offset % noisy_audio_real_lens_val
        offset_adjusted = offset_over_max - tl.math.min(offset_over_max, tl
            .math.max(0, offset_over_max + BLOCK_SIZE_SUM -
            noisy_audio_real_lens_val))
        noisy_audio_block_ptr = offset_adjusted + tl.arange(0, BLOCK_SIZE_SUM)
        noisy_audio_val = tl.load(noisy_audio_ptr + batch_idx *
            noisy_audio_max_len + noisy_audio_block_ptr, mask=
            noisy_audio_block_ptr < noisy_audio_real_lens_val)
        tl.store(output_ptr + batch_idx * clean_audio_max_len +
            clean_audio_block_ptr, clean_audio_vals + noisy_audio_val * (
            relative_rms / noisy_audio_rms), mask=clean_audio_mask)
