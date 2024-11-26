import triton
import triton.language as tl
import torch

@triton.jit
def triton_modulation_scale_shift(x_ptr, modulation_ptr, output_ptr,
    batch_size, head_size, modulation_size, is_mod1, XBLOCK: 'tl.constexpr'):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK + tl.arange(0, XBLOCK)[:]
    batch_idx = xoffset // batch_size
    head_dim_idx = xoffset % head_size
    modulation_offset = head_dim_idx + modulation_size * batch_idx
    x = tl.load(x_ptr + xoffset, None)
    if is_mod1:
        shift = tl.load(modulation_ptr + (modulation_offset + head_size * 0
            ), None, eviction_policy='evict_last')
        scale = tl.load(modulation_ptr + (modulation_offset + head_size * 1
            ), None, eviction_policy='evict_last')
    else:
        shift = tl.load(modulation_ptr + (modulation_offset + head_size * 3
            ), None, eviction_policy='evict_last')
        scale = tl.load(modulation_ptr + (modulation_offset + head_size * 4
            ), None, eviction_policy='evict_last')
    output = (scale + 1.0) * x + shift
    tl.store(output_ptr + xoffset, output, None)
