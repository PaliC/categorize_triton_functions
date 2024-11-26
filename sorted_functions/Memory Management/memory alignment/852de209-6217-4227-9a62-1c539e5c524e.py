import triton
import triton.language as tl
import torch

@triton.jit
def moe_align_block_size_stage4(topk_ids_ptr, sorted_token_ids_ptr,
    expert_ids_ptr, total_tokens_post_pad_ptr, tokens_cnts_ptr, cumsum_ptr,
    num_experts: 'tl.constexpr', block_size: 'tl.constexpr', numel:
    'tl.constexpr', tokens_per_thread: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr'):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)
    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)
    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts
    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)
        ):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)
