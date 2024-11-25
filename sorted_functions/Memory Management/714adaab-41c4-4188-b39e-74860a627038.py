import triton
import triton.language as tl
import torch

@triton.jit
def moe_align_block_size_stage1(topk_ids_ptr, sorted_token_ids_ptr,
    expert_ids_ptr, total_tokens_post_pad_ptr, tokens_cnts_ptr, cumsum_ptr,
    num_experts: 'tl.constexpr', block_size: 'tl.constexpr', numel:
    'tl.constexpr', tokens_per_thread: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr'):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts
    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)
