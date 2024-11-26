import triton
import triton.language as tl
import torch

@triton.jit
def moe_weigths_bp_kernel(dr_ptr, dr_stride_bsk, dr_stride_hout, dk_ptr,
    dk_stride_bsk, dk_stride_hout, dw_ptr, dw_stride_bsk, dw_stride_hout,
    hout: 'tl.constexpr', block_size_hout: 'tl.constexpr'):
    block_idx_bsk = tl.program_id(0)
    block_idx_hout = tl.program_id(1)
    offsets_hout = block_idx_hout * block_size_hout + tl.arange(0,
        block_size_hout)
    block_mask_hout = offsets_hout < hout
    dr_ptrs = (dr_ptr + block_idx_bsk * dr_stride_bsk + offsets_hout *
        dr_stride_hout)
    dk_ptrs = (dk_ptr + block_idx_bsk * dk_stride_bsk + offsets_hout *
        dk_stride_hout)
    dw_ptrs = (dw_ptr + block_idx_bsk * dw_stride_bsk + offsets_hout *
        dw_stride_hout)
    dr = tl.load(dr_ptrs, mask=block_mask_hout, other=0.0)
    dk = tl.load(dk_ptrs, mask=block_mask_hout, other=0.0)
    dw = dr * dk
    tl.store(dw_ptrs, dw, mask=block_mask_hout)
