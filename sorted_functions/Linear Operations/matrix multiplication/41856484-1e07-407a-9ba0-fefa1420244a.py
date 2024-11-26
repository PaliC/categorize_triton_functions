import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(x_ptr, y_ptr, z_ptr, m_size, k_size, n_size, m_block_size:
    'tl.constexpr', k_block_size: 'tl.constexpr', n_block_size: 'tl.constexpr'
    ):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(n_size, n_block_size)
    m_block = pid // num_n_blocks
    n_block = pid % num_n_blocks
    m_offsets = tl.arange(0, m_block_size) + m_block * m_block_size
    n_offsets = tl.arange(0, n_block_size) + n_block * n_block_size
    k_offsets = tl.arange(0, k_block_size)
    x_ptrs = x_ptr + m_offsets[:, None] * k_size + k_offsets[None, :]
    y_ptrs = y_ptr + k_offsets[:, None] * n_size + n_offsets[None, :]
    z_ptrs = z_ptr + m_offsets[:, None] * n_size + n_offsets[None, :]
    z = tl.zeros((m_block_size, n_block_size), tl.float32)
    for _ in range(0, k_size, k_block_size):
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        z += tl.dot(x, y, allow_tf32=False)
        x_ptrs += k_block_size
        y_ptrs += k_block_size * n_size
    tl.store(z_ptrs, z)
