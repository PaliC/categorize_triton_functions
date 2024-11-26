import triton
import triton.language as tl
import torch

@triton.jit
def split_2D_jagged_w_prefix(JaggedIn, DenseSize, OffsetsA, OffsetsB, OutA,
    OutB, D, stride_id, stride_ad, stride_bd, n_prefix_to_B, IS_DENSE_A:
    'tl.constexpr', IS_DENSE_B: 'tl.constexpr', BLOCK_D: 'tl.constexpr',
    IS_REPLACE: 'tl.constexpr'):
    off_z = tl.program_id(1)
    off_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    if IS_REPLACE:
        seq_len = seq_len_a
    else:
        seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return
    if IS_REPLACE:
        seq_start = seq_start_a
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        seq_start = seq_start_a + seq_start_b
        out_seq_b_start = seq_len_a + n_prefix_to_B
    offs_d = tl.arange(0, BLOCK_D)
    in_ptrs = JaggedIn + (seq_start + off_n) * stride_id + offs_d
    if off_n < out_seq_b_start and off_n >= n_prefix_to_B:
        off_a = off_n - n_prefix_to_B
        out_ptrs = OutA + (off_a + seq_start_a) * stride_ad + offs_d
    else:
        off_b = off_n - out_seq_b_start + n_prefix_to_B
        if off_n < n_prefix_to_B:
            off_b += out_seq_b_start - n_prefix_to_B
        out_ptrs = OutB + (off_b + seq_start_b) * stride_bd + offs_d
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)
