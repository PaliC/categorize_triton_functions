import triton
import triton.language as tl
import torch

@triton.jit
def rms_kernel(audios, audios_real_lens, audios_max_len, batch_idx,
    BLOCK_SIZE_RMS: 'tl.constexpr'):
    audios_real_lens_vals = tl.load(audios_real_lens + batch_idx)
    _mean = tl.zeros([BLOCK_SIZE_RMS], dtype=tl.float32)
    for offset in range(0, audios_max_len, BLOCK_SIZE_RMS):
        audios_block_ptr = offset + tl.arange(0, BLOCK_SIZE_RMS)
        audios_mask = audios_block_ptr < audios_real_lens_vals
        audios_vals = tl.load(audios + batch_idx * audios_max_len +
            audios_block_ptr, mask=audios_mask)
        audios_partial_sum_sq = tl.where(audios_mask, tl.math.pow(
            audios_vals, 2.0), 0)
        _mean += audios_partial_sum_sq
    audios_global_sum_sq = tl.sum(_mean, axis=0)
    return tl.sqrt(audios_global_sum_sq / audios_real_lens_vals)
