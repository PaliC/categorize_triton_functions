import triton
import triton.language as tl
import torch

@triton.jit
def flashatt_kernel(q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: 'tl.constexpr',
    B1: 'tl.constexpr'):
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0
    inf = 1000000.0
    q = tl.load(q_ptr + off_i, mask=mask_i)
    exp_sum = tl.zeros((B0,), dtype=tl.float32)
    qk_max = tl.full((B0,), -inf, dtype=tl.float32)
    z = tl.zeros((B0,), dtype=tl.float32)
    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        k = tl.load(k_ptr + off_j, mask=mask_j)
        qk = q[:, None] * k[None, :] + tl.where(mask_ij, 0, -1000000.0)
        new_max = tl.maximum(tl.max(qk, axis=1), qk_max)
        qk_exp = tl.exp2(log2_e * (qk - new_max[:, None]))
        factor = tl.exp2(log2_e * (qk_max - new_max))
        new_exp_sum = exp_sum * factor + tl.sum(qk_exp, axis=1)
        v = tl.load(v_ptr + off_j, mask=mask_j, other=0.0)
        z = z * factor + tl.sum(qk_exp * v[None, :], axis=1)
        qk_max = new_max
        exp_sum = new_exp_sum
    z = z / exp_sum
    tl.store(z_ptr + off_i, z, mask=mask_i)
    return
