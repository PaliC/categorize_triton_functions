import triton
import triton.language as tl
import torch

@triton.jit
def triton_modulation_gate_proj(img_ptr, mod_ptr, proj_ptr, output_ptr,
    batch_size, head_size, modulation_size, XBLOCK: 'tl.constexpr'):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK + tl.arange(0, XBLOCK)[:]
    batch_idx = xoffset // batch_size
    head_dim_idx = xoffset % head_size
    modulation_offset = head_dim_idx + modulation_size * batch_idx
    img = tl.load(img_ptr + xoffset, None)
    mod_gate = tl.load(mod_ptr + (modulation_offset + head_size * 2), None,
        eviction_policy='evict_last')
    proj = tl.load(proj_ptr + xoffset, None)
    output = img + mod_gate * proj
    tl.store(output_ptr + xoffset, output, None)
