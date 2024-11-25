import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_stage1(Q, K_Buffer, sm_scale, Req_to_tokens, B_req_idx,
    B_Start_Loc, B_Seqlen, Att_Out, stride_req_to_tokens_b, stride_qbs,
    stride_qh, stride_buf_kbs, stride_buf_kh, att_stride_h, kv_group_num:
    'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    logit_cap: 'tl.constexpr', Lk: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    reduce_dtype = Att_Out.dtype.element_ty
    cur_kv_head = cur_head // kv_group_num
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)
    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b *
            cur_batch_req_idx + offs_n_new, mask=offs_n_new <
            cur_batch_end_index, other=0)
        offs_buf_k = k_loc[:, None
            ] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[None, :]
        k = tl.load(K_Buffer + offs_buf_k, mask=(offs_n_new[:, None] <
            cur_batch_end_index) & (offs_d[None, :] < Lk), other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        if logit_cap > 0:
            att_value = logit_cap * tanh(att_value / logit_cap)
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index +
            offs_n)
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new <
            cur_batch_end_index)
