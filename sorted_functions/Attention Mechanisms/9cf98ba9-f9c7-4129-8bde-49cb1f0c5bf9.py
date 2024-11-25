import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att2_int8v(Prob, V, V_scale, Out, Req_to_tokens,
    B_req_idx, B_Start_Loc, B_Seqlen, stride_req_to_tokens_b,
    stride_req_to_tokens_s, stride_ph, stride_pbs, stride_vbs, stride_vh,
    stride_vd, stride_vsbs, stride_vsh, stride_vsd, stride_obs, stride_oh,
    stride_od, kv_group_num, BLOCK_DMODEL: 'tl.constexpr', BLOCK_N:
    'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = 0
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    v_loc_off = cur_batch_req_idx * stride_req_to_tokens_b + (
        cur_batch_start_index + offs_n) * stride_req_to_tokens_s
    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n
        ) * stride_pbs
    v_offs = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    vs_offs = cur_kv_head * stride_vsh
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p_value = tl.load(Prob + p_offs + start_n, mask=start_n + offs_n <
            cur_batch_seq_len, other=0.0)
        v_loc = tl.load(Req_to_tokens + v_loc_off + start_n *
            stride_req_to_tokens_s, mask=start_n + offs_n <
            cur_batch_seq_len, other=0.0)
        v_value = tl.load(V + v_offs + v_loc[:, None] * stride_vbs, mask=
            start_n + offs_n[:, None] < cur_batch_seq_len, other=0.0)
        vs_value = tl.load(V_scale + vs_offs + v_loc[:, None] * stride_vsbs,
            mask=start_n + offs_n[:, None] < cur_batch_seq_len, other=0.0)
        acc += tl.sum(p_value[:, None] * v_value * vs_value, 0)
    acc = acc
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return
