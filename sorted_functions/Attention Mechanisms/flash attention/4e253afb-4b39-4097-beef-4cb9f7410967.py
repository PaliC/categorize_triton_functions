import triton
import triton.language as tl
import torch

@triton.jit
def flash_attention_v1_kernel(q_ptr, k_ptr, v_ptr, z_ptr, BN, Lq, Lk, scale,
    H: 'tl.constexpr', dropout_prob=0.0, seed=1337, BLOCK_SIZE_L:
    'tl.constexpr'=64):
    q_ptr += tl.program_id(0) * (Lq * H)
    z_ptr += tl.program_id(0) * (Lq * H)
    k_ptr += tl.program_id(0) * (Lk * H)
    v_ptr += tl.program_id(0) * (Lk * H)
    offs_lq = tl.program_id(1) * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    offs_h = tl.arange(0, H)
    q_mask = offs_lq[:, None] < Lq
    q_offs = offs_lq[:, None] * H + offs_h[None, :]
    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = q
    z = tl.zeros((BLOCK_SIZE_L, H), dtype=tl.float32)
    max_value = tl.zeros((BLOCK_SIZE_L, 1), dtype=tl.float32) + float('-inf')
    denominator = tl.zeros((BLOCK_SIZE_L, 1), dtype=tl.float32)
    for i in range(0, Lk, BLOCK_SIZE_L):
        offs_lk = i + tl.arange(0, BLOCK_SIZE_L)
        kv_mask = offs_lk[:, None] < Lk
        kv_offs = offs_lk[:, None] * H + offs_h[None, :]
        k = tl.load(k_ptr + kv_offs, mask=kv_mask, other=0.0)
        k = k
        qk = tl.dot(q, k.trans(1, 0)) * scale
        qk = tl.where(offs_lq[:, None] >= offs_lk[None, :], qk, float('-inf'))
        block_max_value = tl.max(qk, axis=1, keep_dims=True)
        new_max_value = tl.where(block_max_value > max_value,
            block_max_value, max_value)
        qk = tl.exp(qk - new_max_value)
        multiplier = tl.exp(max_value - new_max_value)
        denominator *= multiplier
        z *= multiplier
        denominator += tl.sum(qk, axis=1, keep_dims=True)
        max_value = new_max_value
        if dropout_prob > 0.0:
            qk_offs = offs_lq[:, None] * Lk + offs_lk[None, :]
            qk = dropout(qk, dropout_prob, seed, qk_offs)
        v = tl.load(v_ptr + kv_offs, mask=kv_mask, other=0.0)
        v = v
        qk = qk
        z = tl.dot(qk, v, acc=z)
    z /= denominator
    z = z
    tl.store(z_ptr + q_offs, z, mask=q_mask)
