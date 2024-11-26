import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_destindex_copy_dequantize_kv(mem_kv_buffer, mem_kv_scale,
    req_to_token_indexs, b_seq_len, b_req_idx, Out, stride_kv_b,
    stride_kv_h, stride_kv_g, stride_kv_d, stride_o_bh, stride_o_l,
    stride_o_g, stride_o_d, stride_s_b, stride_s_h, stride_s_g,
    stride_req_to_tokens_b, stride_req_to_tokens_s, group_size, head_num:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', BLOCK_GROUP_NUM:
    'tl.constexpr', BLOCK_GROUP_DIM: 'tl.constexpr'):
    cur_group = tl.program_id(0)
    start_m = tl.program_id(1)
    cur_bh = tl.program_id(2)
    cur_batch = cur_bh // head_num
    cur_head = cur_bh % head_num
    block_start_loc = BLOCK_SIZE * start_m
    cur_batch_req_idx = tl.load(b_req_idx + cur_batch)
    cur_seq_len = tl.load(b_seq_len + cur_batch)
    offs_kv_loc = block_start_loc + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)
    kv_loc = tl.load(req_to_token_indexs + cur_batch_req_idx *
        stride_req_to_tokens_b + offs_kv_loc, mask=offs_kv_loc < cur_seq_len)
    offs_kv = (kv_loc[:, None] * stride_kv_b + cur_head * stride_kv_h + 
        cur_group * stride_kv_g + offs_d[None, :])
    src_data = tl.load(mem_kv_buffer + offs_kv, mask=offs_kv_loc[:, None] <
        cur_seq_len, other=0.0)
    s_ptrs = (mem_kv_scale + kv_loc * stride_s_b + cur_head * stride_s_h + 
        cur_group * stride_s_g)
    data_scale = tl.load(s_ptrs, mask=offs_kv_loc < cur_seq_len)
    out_data = src_data * data_scale[:, None]
    o_ptrs = Out + cur_bh * stride_o_bh + offs_kv_loc[:, None
        ] * stride_o_l + cur_group * stride_o_g + offs_d[None, :]
    tl.store(o_ptrs, out_data, mask=offs_kv_loc[:, None] < cur_seq_len)
    return
