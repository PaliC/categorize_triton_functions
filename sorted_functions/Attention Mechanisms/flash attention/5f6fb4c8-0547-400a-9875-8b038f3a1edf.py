import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    stride_logic_h, stride_logic_bs, stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od, stride_b_loc_b, stride_b_loc_s,
    other_kv_index, kv_group_num, BLOCK_DMODEL: 'tl.constexpr', BLOCK_N:
    'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_v = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    off_b_loc = cur_batch * stride_b_loc_b + (max_input_len - cur_batch_seq_len
        ) * stride_b_loc_s
    v_ptrs = V + off_v
    e_max = float('-inf')
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(B_Loc + off_b_loc + (start_n + offs_n) *
            stride_b_loc_s, mask=start_n + offs_n < cur_batch_seq_len,
            other=other_kv_index)
        qk = tl.load(Logics + cur_head * stride_logic_h + (
            cur_batch_start_loc + start_n + offs_n) * stride_logic_bs, mask
            =start_n + offs_n < cur_batch_seq_len, other=float('-inf'))
        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_vbs)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max
    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return
