import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_grouped_kernel_stage1(Q, K_Buffer, sm_scale, Req_to_tokens,
    B_req_idx, B_Start_Loc, B_Seqlen, Att_Out, stride_req_to_tokens_b,
    stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh, att_stride_h,
    kv_group_num: 'tl.constexpr', q_head_num: 'tl.constexpr', BLOCK_DMODEL:
    'tl.constexpr', BLOCK_DPE: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    BLOCK_H: 'tl.constexpr', logit_cap: 'tl.constexpr', Lk: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    start_n = tl.program_id(2)
    reduce_dtype = Att_Out.dtype.element_ty
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: 'tl.constexpr' = BLOCK_H
    else:
        VALID_BLOCK_H: 'tl.constexpr' = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[
        None, :]
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = cur_batch * stride_qbs + cur_head[:, None
            ] * stride_qh + offs_dpe[None, :]
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)
    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + offs_q + start_mark, mask=mask_h[:, None] & (offs_d
            [None, :] < Lk))
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b *
            cur_batch_req_idx + offs_n_new, mask=offs_n_new <
            cur_batch_end_index, other=0)
        offs_buf_k = k_loc[None, :
            ] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None]
        k = tl.load(K_Buffer + offs_buf_k, mask=(offs_n_new[None, :] <
            cur_batch_end_index) & (offs_d[:, None] < Lk), other=0.0)
        qk = tl.dot(q, k)
        if BLOCK_DPE > 0:
            qpe = tl.load(Q + off_qpe + start_mark, mask=mask_h[:, None])
            offs_buf_kpe = k_loc[None, :
                ] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_dpe[
                :, None]
            kpe = tl.load(K_Buffer + offs_buf_kpe, mask=offs_n_new[None, :] <
                cur_batch_end_index, other=0.0)
            qk += tl.dot(qpe, kpe)
        qk *= sm_scale
        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)
        offs_o = cur_head[:, None] * att_stride_h + (
            cur_batch_in_all_start_index + offs_n[None, :])
        tl.store(Att_Out + offs_o, qk, mask=mask_h[:, None] & (offs_n_new[
            None, :] < cur_batch_end_index))
