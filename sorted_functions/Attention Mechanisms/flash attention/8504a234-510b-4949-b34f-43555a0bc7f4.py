import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_int8kv(Q, K, V, sm_scale, Out, B_Start_Loc, B_Seqlen,
    b_prompt_cache_len, stride_qbs, stride_qh, stride_qd, stride_kb,
    stride_kh, stride_ks, stride_kd, stride_vb, stride_vh, stride_vs,
    stride_vd, stride_obs, stride_oh, stride_od, kv_group_num, H:
    'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr'):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]
        ) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len,
        cur_batch_seq_len + prompt_cache_len)
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        off_k = cur_batch * stride_kb + (start_n + offs_n[None, :]
            ) * stride_ks + cur_kv_head * stride_kh + offs_d[:, None
            ] * stride_kd
        k = tl.load(K + off_k, mask=start_n + offs_n[None, :] <
            block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        mask = offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :]
        qk = tl.where(mask, qk * sm_scale, -100000000.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = cur_batch * stride_vb + (start_n + offs_n[:, None]
            ) * stride_vs + cur_kv_head * stride_vh + offs_d[None, :
            ] * stride_vd
        v = tl.load(V + off_v, mask=start_n + offs_n[:, None] <
            block_end_loc, other=0.0)
        p = p
        acc = tl.dot(p, v, acc)
        m_i = m_ij
    acc = acc / l_i[:, None]
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]
        ) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
