import triton
import triton.language as tl
import torch

@triton.jit
def rope_kernel_fw(input_ptr, in_seq_len_stride, in_batch_stride,
    output_ptr, cos_ptr, sin_ptr, cos_stride, sin_stride, seq_len, head_dim,
    BLOCK_SIZE: 'tl.constexpr', BATCH_NUM: 'tl.constexpr'):
    pid_seq = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    head_dim_offset = tl.arange(0, BLOCK_SIZE)
    head_dim_mid = head_dim // 2
    mask = head_dim_offset < head_dim_mid
    cos_offset = pid_seq % seq_len * cos_stride + head_dim_offset
    sin_offset = pid_seq % seq_len * sin_stride + head_dim_offset
    cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + sin_offset, mask=mask, other=0.0)
    for batch_idx in tl.static_range(0, BATCH_NUM):
        x1_offset = (pid_seq * in_seq_len_stride + batch_idx *
            in_batch_stride + pid_head * head_dim + head_dim_offset)
        x2_offset = (pid_seq * in_seq_len_stride + batch_idx *
            in_batch_stride + pid_head * head_dim + head_dim_mid +
            head_dim_offset)
        x1 = tl.load(input_ptr + x1_offset, mask=mask, other=0.0)
        x2 = tl.load(input_ptr + x2_offset, mask=mask, other=0.0)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        tl.store(output_ptr + x1_offset, y1, mask=mask)
        tl.store(output_ptr + x2_offset, y2, mask=mask)
    return
