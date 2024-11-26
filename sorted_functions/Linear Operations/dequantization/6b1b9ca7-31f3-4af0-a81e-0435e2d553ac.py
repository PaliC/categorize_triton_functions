import triton
import triton.language as tl
import torch

@triton.jit
def _dequant_kernel(q_idx_ptr, absmax_ptr, qmap_ptr, dq_ptr, stride_qm,
    stride_qn, M, N, GROUP_SIZE: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr'):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rm[:, None] * stride_qm + rn[None, :] * stride_qn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.static_print(offsets)
    group_offsets = offsets // GROUP_SIZE
    tl.static_print('group_offsets', group_offsets)
    q_idx = tl.load(q_idx_ptr + offsets, mask=mask)
    tl.static_print(q_idx)
    q_vals = tl.load(qmap_ptr + q_idx)
    absmax = tl.load(absmax_ptr + group_offsets, mask=group_offsets < M * N //
        GROUP_SIZE)
    dq = q_vals * absmax
    tl.store(dq_ptr + offsets, dq, mask=mask)
