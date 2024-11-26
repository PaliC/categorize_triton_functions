import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(K, Dest_loc, Out, Out_scale,
    stride_k_bs, stride_k_h, stride_k_d, stride_o_bs, stride_o_h,
    stride_o_d, stride_os_bs, stride_os_h, stride_os_d, head_num,
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_HEAD: 'tl.constexpr'):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    dest_index = tl.load(Dest_loc + cur_index)
    src_data = tl.load(K + cur_index * stride_k_bs + offs_h[:, None] *
        stride_k_h + stride_k_d * offs_d[None, :], mask=offs_h[:, None] <
        head_num, other=0.0)
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.0)[:, None]
    q_src_data = src_data / data_scale
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None
        ] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[
        :, None]
    tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < head_num)
    tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < head_num)
