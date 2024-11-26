import triton
import triton.language as tl
import torch

@triton.jit
def triton_bid_merge(y, x, BC: 'tl.constexpr', BT: 'tl.constexpr', d_head:
    'tl.constexpr', n_heads: 'tl.constexpr', batch_size: 'tl.constexpr',
    seq_len: 'tl.constexpr', NT: 'tl.constexpr'):
    i_c = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    batch_idx = i_bh // n_heads
    head_idx = i_bh % n_heads
    block_start_seq = i_t * BT
    block_start_depth = i_c * BC
    seq_range = tl.arange(0, BT)
    depth_range = tl.arange(0, BC)
    seq_idx = block_start_seq + seq_range
    depth_idx = block_start_depth + depth_range
    mask = (seq_idx < seq_len)[:, None] & (depth_idx < d_head)
    offset_normal = (batch_idx * n_heads * seq_len * d_head + head_idx *
        seq_len * d_head + seq_idx[:, None] * d_head + depth_idx)
    offset_mirrored = (batch_idx * n_heads * seq_len * d_head + head_idx *
        seq_len * d_head + (seq_len - seq_idx - 1)[:, None] * d_head +
        depth_idx + batch_size * n_heads * seq_len * d_head)
    normal_vals = tl.load(y + offset_normal, mask=mask)
    mirrored_vals = tl.load(y + offset_mirrored, mask=mask)
    combined_vals = normal_vals + mirrored_vals
    tl.store(x + offset_normal, combined_vals, mask=mask)
