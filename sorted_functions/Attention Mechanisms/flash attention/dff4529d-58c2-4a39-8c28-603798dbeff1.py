import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_int8(Q, K, K_scale, V, V_scale, sm_scale, Req_to_tokens,
    B_req_idx, B_split_start_loc, B_split_ready_cache_len, B_seqlen, Out,
    stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd,
    stride_ksbs, stride_ksh, stride_ksd, stride_vbs, stride_vh, stride_vd,
    stride_vsbs, stride_vsh, stride_vsd, stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s, kv_group_num, BLOCK_M:
    'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_q_split_start_loc = tl.load(B_split_start_loc + cur_batch)
    cur_batch_seq_len = tl.load(B_seqlen + cur_batch)
    cur_batch_seq_start = tl.load(B_split_ready_cache_len + cur_batch)
    cur_batch_q_split_seq_len = cur_batch_seq_len - cur_batch_seq_start
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_q_split_start_loc + offs_m[:, None]
        ) * stride_qbs + cur_head * stride_qh + offs_d[None, :]
    off_k = cur_kv_head * stride_kh + offs_d[:, None]
    off_v = cur_kv_head * stride_vh + offs_d[None, :]
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_q_split_seq_len,
        other=0.0)
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    ks_ptrs = K_scale + cur_kv_head * stride_ksh
    vs_ptrs = V_scale + cur_kv_head * stride_vsh
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(start_m * BLOCK_M < cur_batch_q_split_seq_len, 1, 0)
    for start_n in range(0, block_mask * (cur_batch_seq_start + (start_m + 
        1) * BLOCK_M), BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(Req_to_tokens + cur_batch_req_idx *
            stride_req_to_tokens_b + start_n + offs_n, mask=start_n +
            offs_n < cur_batch_seq_len, other=0)
        k = tl.load(k_ptrs + kv_loc[None, :] * stride_kbs, mask=start_n +
            offs_n[None, :] < cur_batch_seq_len, other=0.0)
        k_scale = tl.load(ks_ptrs + kv_loc[None, :] * stride_ksbs, mask=
            start_n + offs_n[None, :] < cur_batch_seq_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k_scale * k)
        qk *= sm_scale
        qk = tl.where(cur_batch_seq_start + offs_m[:, None] >= start_n +
            offs_n[None, :], qk, float('-100000000.0'))
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + kv_loc[:, None] * stride_vbs, mask=start_n +
            offs_n[:, None] < cur_batch_seq_len, other=0.0)
        v_scale = tl.load(vs_ptrs + kv_loc[:, None] * stride_vsbs, mask=(
            start_n + offs_n)[:, None] < cur_batch_seq_len, other=0.0)
        p = p
        acc += tl.dot(p, v * v_scale)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_q_split_start_loc + offs_m[:, None]
        ) * stride_obs + cur_head * stride_oh + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_q_split_seq_len)
    return
