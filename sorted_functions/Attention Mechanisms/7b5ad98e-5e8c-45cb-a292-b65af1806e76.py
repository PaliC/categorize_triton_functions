import triton
import triton.language as tl
import torch

@triton.jit
def paged_attention_v2(scratchpad_key_ptr, scratchpad_value_ptr,
    partition_buf_ptr, output_ptr, query_ptr, key_cache_ptr,
    value_cache_ptr, block_tables_ptr, context_lens_ptr, scale, num_seqs,
    num_heads, cache_block_stride, num_partitions, PARTITION_SIZE:
    'tl.constexpr', MAX_SEQ_LEN: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr',
    HEAD_SIZE: 'tl.constexpr', MAX_NUM_BLOCKS_PER_SEQ: 'tl.constexpr'):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)
    query_offset = seq_idx * num_seqs + head_idx * HEAD_SIZE
    query_head = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))
    print_tensor_dim(query_head, 'query_head')
    block_table_offset = seq_idx * MAX_NUM_BLOCKS_PER_SEQ
    context_len = tl.load(context_lens_ptr + seq_idx)
    assert context_len <= MAX_SEQ_LEN
    token_start_idx = partition_idx * PARTITION_SIZE
    token_end_idx = min((partition_idx + 1) * PARTITION_SIZE, context_len)
    for tok_idx in range(token_start_idx, token_end_idx):
        logical_block_offset = tok_idx // BLOCK_SIZE
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset +
            logical_block_offset)
        start_of_block_offset = (physical_block_idx * cache_block_stride + 
            head_idx * HEAD_SIZE * BLOCK_SIZE)
        tok_idx_within_block = tok_idx % BLOCK_SIZE
        tok_offsets = start_of_block_offset + BLOCK_SIZE * tl.arange(0,
            HEAD_SIZE) + tok_idx_within_block
        tok_key = tl.load(key_cache_ptr + tok_offsets)
        tok_value = tl.load(value_cache_ptr + tok_offsets)
        scratchpad_offset = seq_idx * (MAX_SEQ_LEN * num_heads * HEAD_SIZE
            ) + tok_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
        print_tensor_dim(scratchpad_key_ptr, 'scratchpad_key_ptr')
        mask = tl.full([HEAD_SIZE], 1, dtype=tl.float32) > 0
        tl.store(scratchpad_key_ptr + scratchpad_offset + tl.arange(0,
            HEAD_SIZE), tok_key, mask)
        tl.store(scratchpad_value_ptr + scratchpad_offset + tl.arange(0,
            HEAD_SIZE), tok_value, mask)
    tl.debug_barrier()
    start_seq_offset = MAX_SEQ_LEN * num_heads * HEAD_SIZE * seq_idx
    start_tok_offsets = start_seq_offset + tl.arange(0, PARTITION_SIZE) * (
        num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    mask = tl.arange(0, PARTITION_SIZE)[:, None] < context_len
    kv_offs = start_tok_offsets[:, None] + tl.arange(0, HEAD_SIZE)[None, :]
    print_tensor_dim(kv_offs, 'kv_offs_v2')
    keys = tl.load(scratchpad_key_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(keys, 'keys_v2')
    scores = tl.sum(scale * keys * query_head[None, :], axis=1)
    print_tensor_dim(keys, 'scores_v2')
    partition_buf_offset = (start_seq_offset + head_idx * HEAD_SIZE + 
        partition_idx * PARTITION_SIZE)
    print_tensor_dim(partition_buf_offset, 'partition_buf_offset_v2')
    tl.store(partition_buf_ptr + partition_buf_offset + tl.arange(0,
        PARTITION_SIZE), scores)
    mask = tl.full([PARTITION_SIZE], -float('inf'), dtype=tl.float32)
    cond = tl.arange(0, PARTITION_SIZE) < context_len
    scores_masked = tl.where(cond, scores, mask)
    scores_minus_max = scores_masked - tl.max(scores_masked, axis=0)
    numerator = tl.exp(scores_minus_max)
    denominator = tl.sum(numerator, axis=0) + float(1e-06)
    logits = numerator / denominator
    print_tensor_dim(logits, 'logits_v2')
    values = tl.load(scratchpad_value_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(values, 'values_v2')
    weighted_values += tl.sum(values * logits[:, None], axis=0)
    print_tensor_dim(weighted_values, 'weighed_values_v2')
    output_offset = seq_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE),
        weighted_values)
