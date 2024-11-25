import triton
import triton.language as tl
import torch

@triton.jit
def _sparse_fwd_kernel_flash_decode_stage1(Q_Label, K_Label_Buffer,
    sm_scale, Req_to_tokens, B_Seqlen, Att_Out, stride_req_to_tokens_b,
    stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh, att_stride_h,
    att_stride_b, kv_group_num: 'tl.constexpr', BLOCK_DMODEL:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', logit_cap: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len
    min_val = -float('inf')
    att_value = tl.full([BLOCK_N], min_val, dtype=tl.float32)
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    block_index = start_n * BLOCK_N
    block_mask = tl.where(block_index < cur_batch_seq_len, 1, 0)
    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q_Label + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch +
            offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        offs_buf_k = k_loc[:, None
            ] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[None, :]
        k = tl.load(K_Label_Buffer + offs_buf_k, mask=offs_n_new[:, None] <
            cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        if logit_cap > 0:
            att_value = logit_cap * tanh(att_value / logit_cap)
    att_value = tl.where(offs_n < cur_batch_end_index, att_value, min_val)
    off_o = cur_head * att_stride_h + (cur_batch * att_stride_b + offs_n)
    tl.store(Att_Out + off_o, att_value)
