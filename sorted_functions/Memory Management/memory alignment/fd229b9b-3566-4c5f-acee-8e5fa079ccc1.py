import triton
import triton.language as tl
import torch

@triton.jit
def moe_align_block_size_stage3(topk_ids_ptr, sorted_token_ids_ptr,
    expert_ids_ptr, total_tokens_post_pad_ptr, tokens_cnts_ptr, cumsum_ptr,
    num_experts: 'tl.constexpr', block_size: 'tl.constexpr', numel:
    'tl.constexpr', tokens_per_thread: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr'):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)
