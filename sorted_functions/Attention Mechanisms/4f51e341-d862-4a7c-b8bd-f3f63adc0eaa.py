import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_grouped_kernel_stage2(logits, V_Buffer, Out, Req_to_tokens,
    B_req_idx, B_Start_Loc, B_Seqlen, stride_logic_h, stride_buf_vbs,
    stride_buf_vh, stride_obs, stride_oh, stride_req_to_token_b,
    kv_group_num: 'tl.constexpr', q_head_num: 'tl.constexpr', BLOCK_DMODEL:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_H: 'tl.constexpr', Lv:
    'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: 'tl.constexpr' = BLOCK_H
    else:
        VALID_BLOCK_H: 'tl.constexpr' = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_buf_v = cur_kv_head * stride_buf_vh + offs_d[None, :]
    v_ptrs = V_Buffer + offs_buf_v
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(Req_to_tokens + cur_batch_req_idx *
            stride_req_to_token_b + (start_n + offs_n), mask=start_n +
            offs_n < cur_batch_seq_len, other=0)
        offs_qk = cur_head[:, None] * stride_logic_h + (cur_batch_start_loc +
            start_n + offs_n[None, :])
        qk = tl.load(logits + offs_qk, mask=mask_h[:, None] & (start_n +
            offs_n[None, :] < cur_batch_seq_len), other=float('-inf'))
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        e_sum = e_sum * old_scale + tl.sum(p, 1)
        v = tl.load(v_ptrs + v_index[:, None] * stride_buf_vbs, mask=offs_d
            [None, :] < Lv)
        p = p
        acc = acc * old_scale[:, None] + tl.dot(p, v)
        e_max = n_e_max
    acc = acc / e_sum[:, None]
    off_o = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_d[
        None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=mask_h[:, None] & (offs_d[None, :] < Lv))
