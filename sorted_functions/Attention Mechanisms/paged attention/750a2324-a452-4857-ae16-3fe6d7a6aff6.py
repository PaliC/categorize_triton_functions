import triton
import triton.language as tl
import torch

@triton.jit
def paged_attention_v1(scratchpad_key_ptr, scratchpad_value_ptr, output_ptr,
    query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr,
    context_lens_ptr, scale, num_seqs, num_heads, cache_block_stride,
    MAX_SEQ_LEN: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', HEAD_SIZE:
    'tl.constexpr', MAX_NUM_BLOCKS_PER_SEQ: 'tl.constexpr'):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    query_offset = seq_idx * num_seqs + head_idx * HEAD_SIZE
    query_head = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))
    block_table_offset = seq_idx * MAX_NUM_BLOCKS_PER_SEQ
    context_len = tl.load(context_lens_ptr + seq_idx)
    for tok_idx in range(0, context_len):
        logical_block_idx = tok_idx // BLOCK_SIZE
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset +
            logical_block_idx)
        start_of_block_offset = (physical_block_idx * cache_block_stride + 
            head_idx * HEAD_SIZE * BLOCK_SIZE)
        tok_idx_within_block = tok_idx % BLOCK_SIZE
        tok_offsets = start_of_block_offset + BLOCK_SIZE * tl.arange(0,
            HEAD_SIZE) + tok_idx_within_block
        tok_key = tl.load(key_cache_ptr + tok_offsets)
        tok_value = tl.load(value_cache_ptr + tok_offsets)
        scratchpad_offset = seq_idx * (MAX_SEQ_LEN * num_heads * HEAD_SIZE
            ) + tok_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
        tl.store(scratchpad_key_ptr + scratchpad_offset + tl.arange(0,
            HEAD_SIZE), tok_key)
        tl.store(scratchpad_value_ptr + scratchpad_offset + tl.arange(0,
            HEAD_SIZE), tok_value)
    tl.debug_barrier()
    start_seq_offset = MAX_SEQ_LEN * num_heads * HEAD_SIZE * seq_idx
    start_tok_offset = start_seq_offset + tl.arange(0, MAX_SEQ_LEN) * (
        num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    mask = tl.arange(0, MAX_SEQ_LEN)[:, None] < context_len
    kv_offs = start_tok_offset[:, None] + tl.arange(0, HEAD_SIZE)[None, :]
    print_tensor_dim(kv_offs, 'kv_offs_v1')
    keys = tl.load(scratchpad_key_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(keys, 'keys_v1')
    values = tl.load(scratchpad_value_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(values, 'values_v1')
    scores = tl.sum(scale * keys * query_head[None, :], axis=1)
    mask = tl.full([MAX_SEQ_LEN], -float('inf'), dtype=tl.float32)
    cond = tl.arange(0, MAX_SEQ_LEN) < context_len
    scores_masked = tl.where(cond, scores, mask)
    scores_minus_max = scores_masked - tl.max(scores_masked, axis=0)
    numerator = tl.exp(scores_minus_max)
    denominator = tl.sum(numerator, axis=0) + float(1e-06)
    logits = numerator / denominator
    print_tensor_dim(logits, 'logits_v1')
    weighted_values = tl.sum(values * logits[:, None], axis=0)
    print_tensor_dim(weighted_values, 'weighted_values_v1')
    output_offset = seq_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE),
        weighted_values)
