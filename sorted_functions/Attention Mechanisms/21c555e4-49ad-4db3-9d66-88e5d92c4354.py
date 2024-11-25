import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, Out, stride_qbs,
    stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh, stride_obs,
    stride_oh, kv_group_num: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', IS_CAUSAL:
    'tl.constexpr', Lk: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]
        ) * stride_qbs + cur_head * stride_qh + offs_d[None, :]
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[
        :, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[
        None, :]
    mask_d = offs_d < Lk
    q = tl.load(Q + off_q, mask=(offs_m[:, None] < cur_batch_seq_len) &
        mask_d[None, :], other=0.0)
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    end_n = cur_batch_seq_len if not IS_CAUSAL else tl.minimum((start_m + 1
        ) * BLOCK_M, cur_batch_seq_len)
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) *
            stride_kbs, mask=(start_n + offs_n[None, :] < cur_batch_seq_len
            ) & mask_d[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        if IS_CAUSAL:
            qk += tl.where((start_n + offs_n[None, :] < cur_batch_seq_len) &
                (offs_m[:, None] >= start_n + offs_n[None, :]), 0, float(
                '-inf'))
        else:
            qk += tl.where(start_n + offs_n[None, :] < cur_batch_seq_len, 0,
                float('-inf'))
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
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) *
            stride_vbs, mask=(start_n + offs_n[:, None] < cur_batch_seq_len
            ) & mask_d[None, :], other=0.0)
        p = p
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]
        ) * stride_obs + cur_head * stride_oh + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) &
        mask_d[None, :])
