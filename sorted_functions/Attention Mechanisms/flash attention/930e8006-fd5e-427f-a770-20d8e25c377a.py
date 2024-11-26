import triton
import triton.language as tl
import torch

@triton.jit
def flash_attention_v1_kernel(q_ptr, k_ptr, v_ptr, o_ptr, q_batch_stride,
    q_heads_stride, q_seq_stride, q_dim_stride, k_batch_stride,
    k_heads_stride, k_seq_stride, k_dim_stride, v_batch_stride,
    v_heads_stride, v_seq_stride, v_dim_stride, out_batch_stride,
    out_heads_stride, out_seq_stride, out_dim_stride, num_kv_groups,
    n_heads, m_size, n_size, BLOCK_DHEAD_SIZE: 'tl.constexpr', BLOCK_M_SIZE:
    'tl.constexpr', BLOCK_N_SIZE: 'tl.constexpr', sm_scale, causal_mask):
    """
    flashattention 内核实现
    """
    block_m_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    cur_batch_idx = head_idx // n_heads
    cur_head_idx = head_idx % n_heads
    cur_kv_head_idx = cur_head_idx // num_kv_groups
    m_range_offs = tl.arange(0, BLOCK_M_SIZE)
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)
    dhead_range_offs = tl.arange(0, BLOCK_DHEAD_SIZE)
    m_offs = block_m_idx * BLOCK_M_SIZE + m_range_offs
    q_offs = cur_batch_idx * q_batch_stride + cur_head_idx * q_heads_stride + (
        m_offs[:, None] * q_seq_stride + dhead_range_offs[None, :] *
        q_dim_stride)
    k_offs = (cur_batch_idx * k_batch_stride + cur_kv_head_idx *
        k_heads_stride + (n_range_offs[:, None] * k_seq_stride + 
        dhead_range_offs[None, :] * k_dim_stride))
    v_offs = (cur_batch_idx * v_batch_stride + cur_kv_head_idx *
        v_heads_stride + (n_range_offs[:, None] * v_seq_stride + 
        dhead_range_offs[None, :] * v_dim_stride))
    o_offs = (cur_batch_idx * out_batch_stride + cur_head_idx *
        out_heads_stride + (m_offs[:, None] * out_seq_stride + 
        dhead_range_offs[None, :] * out_dim_stride))
    q_ptrs = q_ptr + q_offs
    k_ptrs = k_ptr + k_offs
    v_ptrs = v_ptr + v_offs
    out_ptrs = o_ptr + o_offs
    l_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float('inf')
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)
    q_mask = m_offs[:, None] < m_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE):
        block_n_offs = block_n_start_idx + n_range_offs
        k_mask = block_n_offs[:, None] < n_size
        k = tl.load(k_ptrs + block_n_start_idx * k_seq_stride, mask=k_mask,
            other=0.0)
        qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        if causal_mask:
            offs_k = block_n_offs
            offs_m = m_offs
            mask = offs_m[:, None] >= offs_k[None, :]
            qk = tl.where(mask, qk * sm_scale, -100000000.0)
        else:
            qk = qk * sm_scale
        l_j = tl.max(qk, 1)
        numerators = tl.exp(qk - l_j[:, None])
        d_j = tl.sum(numerators, 1)
        l_new = tl.maximum(l_i, l_j)
        alpha = tl.exp(l_i - l_new)
        beta = tl.exp(l_j - l_new)
        d_new = alpha * d_i + beta * d_j
        p_scale = beta / d_new
        p = numerators * p_scale[:, None]
        sigma = d_i / d_new * alpha
        acc = acc * sigma[:, None]
        v = tl.load(v_ptrs + block_n_start_idx * v_seq_stride, mask=k_mask,
            other=0.0)
        p = p
        acc += tl.dot(p, v)
        l_i = l_new
        d_i = d_new
    out_mask = m_offs[:, None] < m_size
    tl.store(out_ptrs, acc, mask=out_mask)
