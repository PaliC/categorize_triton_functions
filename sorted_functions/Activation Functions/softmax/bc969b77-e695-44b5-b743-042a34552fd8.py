import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel_brute_force(x_ptr, z_ptr, N0, N1, T, B0: 'tl.constexpr',
    B1: 'tl.constexpr'):
    """3 loops ver."""
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0
    exp_sum = tl.zeros([B0], dtype=tl.float32)
    x_max = tl.zeros([B0], dtype=tl.float32)
    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        x_max = tl.maximum(x_max, tl.max(x, axis=1))
    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        exp_x = tl.exp2(log2_e * (x - x_max[:, None]))
        exp_sum += tl.sum(exp_x, axis=1)
    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        exp_x = tl.exp2(log2_e * (x - x_max[:, None]))
        z = exp_x / exp_sum[:, None]
        tl.store(z_ptr + off_ij, z, mask=mask_ij)
    return
