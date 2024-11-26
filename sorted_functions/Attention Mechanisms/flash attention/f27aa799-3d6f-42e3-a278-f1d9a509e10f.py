import triton
import triton.language as tl
import torch

@triton.jit
def _rope_fwd(q_ptr, k_ptr, f_ptr, oq_ptr, ok_ptr, stride, d, BLOCK_SIZE:
    'tl.constexpr'):
    bh_idx = tl.program_id(0)
    s_idx = tl.program_id(1)
    q_start_ptr = q_ptr + bh_idx * stride
    k_start_ptr = k_ptr + bh_idx * stride
    oq_start_ptr = oq_ptr + bh_idx * stride
    ok_start_ptr = ok_ptr + bh_idx * stride
    d_half = d // 2
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_offsets2 = tl.arange(0, BLOCK_SIZE * 2)
    f0_ptrs = f_ptr + s_idx * d * 2 + col_offsets2 * 2
    f1_ptrs = f_ptr + s_idx * d * 2 + col_offsets2 * 2 + 1
    f0 = tl.load(f0_ptrs, mask=col_offsets2 < d, other=0.0).reshape(BLOCK_SIZE,
        2)
    f1 = tl.load(f1_ptrs, mask=col_offsets2 < d, other=0.0).reshape(BLOCK_SIZE,
        2)
    q0_ptrs = q_start_ptr + s_idx * d + col_offsets * 2
    q1_ptrs = q_start_ptr + s_idx * d + col_offsets * 2 + 1
    q0 = tl.load(q0_ptrs, mask=col_offsets < d_half, other=0.0).reshape(
        BLOCK_SIZE, 1)
    q1 = tl.load(q1_ptrs, mask=col_offsets < d_half, other=0.0).reshape(
        BLOCK_SIZE, 1)
    k0_ptrs = k_start_ptr + s_idx * d + col_offsets * 2
    k1_ptrs = k_start_ptr + s_idx * d + col_offsets * 2 + 1
    k0 = tl.load(k0_ptrs, mask=col_offsets < d_half, other=0.0).reshape(
        BLOCK_SIZE, 1)
    k1 = tl.load(k1_ptrs, mask=col_offsets < d_half, other=0.0).reshape(
        BLOCK_SIZE, 1)
    oq = f0 * q0 + f1 * q1
    ok = f0 * k0 + f1 * k1
    oq_ptrs = oq_start_ptr + s_idx * d + col_offsets2
    ok_ptrs = ok_start_ptr + s_idx * d + col_offsets2
    tl.store(oq_ptrs, oq.reshape(BLOCK_SIZE * 2), mask=col_offsets2 < d)
    tl.store(ok_ptrs, ok.reshape(BLOCK_SIZE * 2), mask=col_offsets2 < d)
