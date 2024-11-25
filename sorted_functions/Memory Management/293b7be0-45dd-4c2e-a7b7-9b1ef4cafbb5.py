import triton
import triton.language as tl
import torch

@triton.jit
def moe_align_block_size_stage2(topk_ids_ptr, sorted_token_ids_ptr,
    expert_ids_ptr, total_tokens_post_pad_ptr, tokens_cnts_ptr, cumsum_ptr,
    num_experts: 'tl.constexpr', block_size: 'tl.constexpr', numel:
    'tl.constexpr', tokens_per_thread: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr'):
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)
