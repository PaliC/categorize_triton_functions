import triton
import triton.language as tl
import torch

@triton.heuristics({'BACKWARD_PASS': lambda args: bool(args['BACKWARD_PASS'])})
@triton.jit
def _rope_embedding(Q, Q_row_stride, cos, cos_row_stride, sin,
    sin_row_stride, seqlen, head_dim: 'tl.constexpr', n_heads:
    'tl.constexpr', BACKWARD_PASS: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    ROPE_GROUP_SIZE = 4
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim
    sin1 = tl.load(sin + row_position % seqlen * sin_row_stride + 
        half_head_dim * 0 + col_offsets, mask=mask, other=0)
    cos1 = tl.load(cos + row_position % seqlen * cos_row_stride + 
        half_head_dim * 0 + col_offsets, mask=mask, other=0)
    if BACKWARD_PASS:
        sin1 = -sin1
    pass
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min(head_start + ROPE_GROUP_SIZE, n_heads)
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = (row_position * Q_row_stride + k * head_dim + col_offsets +
            half_head_dim)
        Q1 = tl.load(Q + offs_q1, mask=mask, other=0)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0)
        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)
    pass
