import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_destindex_copy_quantize_int4_kv(K, Dest_loc, Out, Out_scale,
    stride_k_bs, stride_k_h, stride_k_g, stride_k_d, stride_o_bs,
    stride_o_h, stride_o_g, stride_o_d, stride_os_bs, stride_os_h,
    stride_os_g, group_size, BLOCK_GROUP_NUM: 'tl.constexpr',
    BLOCK_GROUP_DIM: 'tl.constexpr'):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM // 2)
    dest_index = tl.load(Dest_loc + cur_index)
    src_data_0 = tl.load(K + cur_index * stride_k_bs + cur_head *
        stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * 2,
        mask=offs_g[:, None] < group_size, other=0.0)
    src_data_1 = tl.load(K + cur_index * stride_k_bs + cur_head *
        stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * 2 + 1,
        mask=offs_g[:, None] < group_size, other=0.0)
    abs_data_0 = tl.abs(src_data_0)
    abs_data_1 = tl.abs(src_data_1)
    data_scale = tl.maximum(tl.max(abs_data_0, axis=1), tl.max(abs_data_1,
        axis=1)) / 7.0
    q_src_data_0 = src_data_0 / data_scale[:, None]
    q_src_data_0 = tl.where(q_src_data_0 > 7, 7, q_src_data_0)
    q_src_data_0 = tl.where(q_src_data_0 < -7, -7, q_src_data_0)
    q_src_data_1 = src_data_1 / data_scale[:, None]
    q_src_data_1 = tl.where(q_src_data_1 > 7, 7, q_src_data_1)
    q_src_data_1 = tl.where(q_src_data_1 < -7, -7, q_src_data_1)
    low_4 = (q_src_data_0 & 128) >> 4 | q_src_data_0 & 15
    high_4 = ((q_src_data_1 & 128) >> 4 | q_src_data_1 & 15) << 4
    out_data = low_4 | high_4
    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[
        :, None] * stride_o_g + offs_d[None, :]
    os_ptrs = (Out_scale + dest_index * stride_os_bs + cur_head *
        stride_os_h + offs_g)
    tl.store(o_ptrs, out_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
    return
