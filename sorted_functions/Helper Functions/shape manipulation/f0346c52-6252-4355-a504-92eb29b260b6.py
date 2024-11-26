import triton
import triton.language as tl
import torch

@triton.jit
def _rotary_kernel(Q, Cos, Sin, stride_qbs, stride_qh, stride_qd,
    stride_cosbs, stride_cosd, stride_sinbs, stride_sind, max_total_len, H,
    BLOCK_HEAD: 'tl.constexpr', BLOCK_SEQ: 'tl.constexpr', BLOCK_DMODEL:
    'tl.constexpr'):
    cur_head_index = tl.program_id(0)
    cur_seq_index = tl.program_id(1)
    cur_head_range = cur_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_range0 = tl.arange(0, BLOCK_DMODEL // 2)
    dim_range1 = tl.arange(BLOCK_DMODEL // 2, BLOCK_DMODEL)
    off_q0 = cur_seq_range[:, None, None] * stride_qbs + cur_head_range[
        None, :, None] * stride_qh + dim_range0[None, None, :] * stride_qd
    off_q1 = cur_seq_range[:, None, None] * stride_qbs + cur_head_range[
        None, :, None] * stride_qh + dim_range1[None, None, :] * stride_qd
    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[
        None, None, :] * stride_cosd
    q0 = tl.load(Q + off_q0, mask=(cur_seq_range[:, None, None] <
        max_total_len) & (cur_head_range[None, :, None] < H), other=0.0)
    q1 = tl.load(Q + off_q1, mask=(cur_seq_range[:, None, None] <
        max_total_len) & (cur_head_range[None, :, None] < H), other=0.0)
    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] <
        max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] <
        max_total_len, other=0.0)
    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos
    tl.store(Q + off_q0, out0, mask=(cur_seq_range[:, None, None] <
        max_total_len) & (cur_head_range[None, :, None] < H))
    tl.store(Q + off_q1, out1, mask=(cur_seq_range[:, None, None] <
        max_total_len) & (cur_head_range[None, :, None] < H))
    return
