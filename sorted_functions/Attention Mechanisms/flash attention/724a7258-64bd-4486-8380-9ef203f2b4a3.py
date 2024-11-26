import triton
import triton.language as tl
import torch

@triton.jit
def flash_attention_v2_kernel(q_ptr, k_ptr, v_ptr, o_ptr, q_batch_stride,
    q_heads_stride, q_seq_stride, q_dim_stride, k_batch_stride,
    k_heads_stride, k_seq_stride, k_dim_stride, v_batch_stride,
    v_heads_stride, v_seq_stride, v_dim_stride, out_batch_stride,
    out_heads_stride, out_seq_stride, out_dim_stride, num_kv_groups,
    n_heads, m_size, n_size, HEAD_DIM: 'tl.constexpr', BLOCK_M_SIZE:
    'tl.constexpr', BLOCK_N_SIZE: 'tl.constexpr', qk_scale, causal_mask):
    """
    flashattention2 内核实现
    """
    block_m_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    cur_batch_idx = head_idx // n_heads
    cur_head_idx = head_idx % n_heads
    cur_kv_head_idx = cur_head_idx // num_kv_groups
    m_range_offs = tl.arange(0, BLOCK_M_SIZE)
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)
    dhead_range_offs = tl.arange(0, HEAD_DIM)
    offs_m = block_m_idx * BLOCK_M_SIZE + m_range_offs
    offs_q = cur_batch_idx * q_batch_stride + cur_head_idx * q_heads_stride + (
        offs_m[:, None] * q_seq_stride + dhead_range_offs[None, :] *
        q_dim_stride)
    offs_k = (cur_batch_idx * k_batch_stride + cur_kv_head_idx *
        k_heads_stride + (n_range_offs[:, None] * k_seq_stride + 
        dhead_range_offs[None, :] * k_dim_stride))
    offs_v = (cur_batch_idx * v_batch_stride + cur_kv_head_idx *
        v_heads_stride + (n_range_offs[:, None] * v_seq_stride + 
        dhead_range_offs[None, :] * v_dim_stride))
    offs_o = (cur_batch_idx * out_batch_stride + cur_head_idx *
        out_heads_stride + (offs_m[:, None] * out_seq_stride + 
        dhead_range_offs[None, :] * out_dim_stride))
    q_ptrs = q_ptr + offs_q
    k_ptrs = k_ptr + offs_k
    v_ptrs = v_ptr + offs_v
    out_ptrs = o_ptr + offs_o
    q_mask = offs_m[:, None] < m_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    m_i = tl.zeros([BLOCK_M_SIZE], dtype=tl.float32) - float('inf')
    d_i = tl.zeros([BLOCK_M_SIZE], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M_SIZE, HEAD_DIM], dtype=tl.float32)
    acc, d_i = _attn_fwd_inner(acc, m_i, d_i, q, k_ptrs, v_ptrs,
        k_seq_stride, v_seq_stride, offs_m, qk_scale, n_size, causal_mask,
        BLOCK_M_SIZE, BLOCK_N_SIZE, v_ptr.dtype.element_ty == tl.float8e5)
    acc = acc / d_i[:, None]
    out_mask = offs_m[:, None] < m_size
    tl.store(out_ptrs, acc, mask=out_mask)
