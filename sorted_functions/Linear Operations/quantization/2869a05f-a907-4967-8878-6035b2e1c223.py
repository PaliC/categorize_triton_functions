import triton
import triton.language as tl
import torch

@triton.jit
def quant_kernel(src_ptr, stride_srcb, stride_srcm, stride_srcn, dst_ptr,
    stride_dstb, stride_dstm, stride_dstn, output_scale, B, M:
    'tl.constexpr', N: 'tl.constexpr', np2_M: 'tl.constexpr', np2_N:
    'tl.constexpr'):
    """
    quant fp16 tensor to int4
    """
    batch_id = tl.program_id(axis=0) + tl.program_id(axis=1) * tl.num_programs(
        axis=0)
    index_rows = tl.arange(0, np2_M)
    index_cols = tl.arange(0, np2_N)
    src_ptrs = src_ptr + batch_id * stride_srcb + index_rows[:, None
        ] * stride_srcm + index_cols[None, :] * stride_srcn
    src_mask = (index_rows[:, None] < M) & (index_cols[None, :] < N)
    src = tl.load(src_ptrs, mask=src_mask, other=0.0)
    abs_src_val = tl.abs(src)
    max_src_val = tl.max(abs_src_val)
    scale = max_src_val / 7.0
    quant_val = libdevice.llrint(src / scale)
    quant_val = max(-8, min(quant_val, 7))
    quant_val = quant_val.reshape(np2_M, np2_N // 2, 2, can_reorder=False)
    quant_val_even, quant_val_odd = quant_val.split()
    quant_val_odd = quant_val_odd << 4
    res = tl.zeros((np2_M, np2_N // 2), dtype=tl.uint8)
    res = res | quant_val_odd & 240
    res = res | quant_val_even & 15
    offs_resm = tl.arange(0, np2_M)
    offs_resn = tl.arange(0, np2_N // 2)
    dst_ptrs = dst_ptr + stride_dstb * batch_id + stride_dstm * offs_resm[:,
        None] + stride_dstn * offs_resn[None, :]
    res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N // 2)
    tl.store(dst_ptrs, res, mask=res_mask)
    tl.store(output_scale + batch_id, scale)
