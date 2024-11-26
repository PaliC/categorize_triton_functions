import triton
import triton.language as tl
import torch

@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: 'tl.constexpr', B1: 'tl.constexpr'
    ):
    block_id_i = tl.program_id(0)
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0
    z = tl.zeros([B0], dtype=tl.float32)
    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        z += tl.sum(x, axis=1)
    tl.store(z_ptr + off_i, z, mask=mask_i)
    return
