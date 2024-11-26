import triton
import triton.language as tl
import torch

@triton.jit
def sum_kernel(x_ptr, output_ptr, M, N, BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr'):
    pid_m = tl.program_id(axis=0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offset < M
    out = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        n_offset = start + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        n_mask = n_offset < N
        mask = m_mask[:, None] & n_mask[None, :]
        inp = tl.load(x_ptr + offset, mask=mask, other=0)
        out += tl.sum(inp, axis=1)
    tl.store(output_ptr + m_offset, out, mask=m_mask)
