import triton
import triton.language as tl
import torch

@triton.jit
def MSABwdFused(b_ij_ptr, logsumexp_ptr, N_head, RES_LEN: 'tl.constexpr',
    BLOCK_SIZE_ROW: 'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr'):
    pid_zh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_z = pid_zh // N_head
    pid_h = pid_zh % N_head
    log2_e = 1.44269504089
    z_off = pid_z
    h_off = pid_h
    i_off = pid_i * BLOCK_SIZE_ROW
    offs_i = i_off + tl.arange(0, BLOCK_SIZE_ROW)
    lse_off = z_off * RES_LEN * N_head + offs_i[:, None] * N_head + h_off
    lse_mask = (offs_i < RES_LEN)[:, None]
    logsumexp = tl.load(logsumexp_ptr + lse_off, lse_mask, 0)
    for j in range(0, RES_LEN, BLOCK_SIZE_COL):
        offs_j = j + tl.arange(0, BLOCK_SIZE_COL)
        b_offs = z_off * RES_LEN * RES_LEN * N_head + offs_i[:, None
            ] * RES_LEN * N_head + offs_j[None, :] * N_head + h_off
        ij_mask = (offs_i < RES_LEN)[:, None] & (offs_j < RES_LEN)[None, :]
        b = tl.load(b_ij_ptr + b_offs, ij_mask, -INF)
        b = tl.exp2(log2_e * (b - logsumexp))
        tl.store(b_ij_ptr + b_offs, b, ij_mask)
