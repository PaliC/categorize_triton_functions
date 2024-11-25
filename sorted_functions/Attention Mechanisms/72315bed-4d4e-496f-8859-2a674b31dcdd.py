import triton
import triton.language as tl
import torch

@triton.jit
def _paged_attn_v2_reduce_kernel(out_ptr, m_i_ptr, l_i_ptr, tmp_out_ptr,
    context_lens_ptr, max_num_partitions, stride_o0, stride_o1, stride_o2,
    HEAD_SIZE: 'tl.constexpr', QUERY_GROUP_SIZE: 'tl.constexpr',
    NUM_KV_HEADS: 'tl.constexpr', PARTITION_SIZE: 'tl.constexpr',
    NUM_PARTITIONS: 'tl.constexpr'):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    context_len = tl.load(context_lens_ptr + seq_idx)
    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)
    group_head_offset = tl.arange(0, QUERY_GROUP_SIZE)[:, None
        ] * HEAD_SIZE + tl.arange(0, HEAD_SIZE)[None, :]
    if num_partitions == 1:
        tmp_out_offset = ((seq_idx * NUM_KV_HEADS + kv_head_idx) *
            max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE +
            group_head_offset)
        tmp_out = tl.load(tmp_out_ptr + tmp_out_offset)
        out_offset = (seq_idx * stride_o0 + kv_head_idx * QUERY_GROUP_SIZE *
            stride_o1 + group_head_offset * stride_o2)
        tl.store(out_ptr + out_offset, tmp_out)
        return
    ml_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx
        ) * max_num_partitions * QUERY_GROUP_SIZE + tl.arange(0, NUM_PARTITIONS
        )[:, None] * QUERY_GROUP_SIZE + tl.arange(0, QUERY_GROUP_SIZE)[None, :]
    mask = tl.arange(0, NUM_PARTITIONS)[:, None] < num_partitions
    m_i = tl.load(m_i_ptr + ml_offset, mask=mask, other=float('-inf'))
    m = tl.max(m_i, axis=0)
    l_i = tl.load(l_i_ptr + ml_offset, mask=mask, other=0.0)
    l_i *= tl.exp(m_i - m[None, :])
    l = tl.sum(l_i, axis=0)
    r = l_i / l[None, :]
    r = tl.reshape(r, (NUM_PARTITIONS, QUERY_GROUP_SIZE, 1))
    tmp_out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx
        ) * max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE + tl.arange(0,
        NUM_PARTITIONS)[:, None, None
        ] * QUERY_GROUP_SIZE * HEAD_SIZE + tl.arange(0, QUERY_GROUP_SIZE)[
        None, :, None] * HEAD_SIZE + tl.arange(0, HEAD_SIZE)[None, None, :]
    tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=mask[:, :, None],
        other=0.0)
    out = tl.sum(tmp_out * r, axis=0)
    out_offset = (seq_idx * stride_o0 + kv_head_idx * QUERY_GROUP_SIZE *
        stride_o1 + group_head_offset * stride_o2)
    tl.store(out_ptr + out_offset, out)
