import triton
import triton.language as tl
import torch

@triton.jit
def triton_batch_lora_B(output, x, w, a_start, a_len, a_loc, batch_req_bins,
    a_scaling, qkvo_offset: 'tl.constexpr', NUM_TOKENS: 'tl.constexpr',
    HIDDEN: 'tl.constexpr', MAX_LORA_RANK: 'tl.constexpr', BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'
    ):
    return
