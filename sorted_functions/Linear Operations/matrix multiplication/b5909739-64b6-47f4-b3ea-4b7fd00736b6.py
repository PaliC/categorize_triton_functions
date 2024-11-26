import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kv_parallel(K, V, S, KV, b: 'tl.constexpr', h: 'tl.constexpr', n:
    'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr', BLOCK:
    'tl.constexpr', NUM_BLOCK: 'tl.constexpr', D_FBLOCK: 'tl.constexpr',
    E_FBLOCK: 'tl.constexpr', NUM_FBLOCK: 'tl.constexpr', CBLOCK:
    'tl.constexpr', NUM_CBLOCK: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)
    off_h = off_bh % h
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK
    block_offset = off_block * BLOCK
    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e
    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK
    K_trans_block_ptr = K + k_offset + k_block_offset + d_offset + tl.arange(
        0, CBLOCK)[None, :] * d + tl.arange(0, D_FBLOCK)[:, None]
    V_block_ptr = V + v_offset + v_block_offset + e_offset + tl.arange(0,
        CBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    KV_block_ptr = (KV + kv_offset + kv_block_offset + d_offset * e +
        e_offset + tl.arange(0, D_FBLOCK)[:, None] * e + tl.arange(0,
        E_FBLOCK)[None, :])
    s_ptrs = S + off_h
    s = tl.load(s_ptrs)
    c_array = tl.arange(0, CBLOCK)
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for j in range(NUM_CBLOCK):
        k_trans = tl.load(K_trans_block_ptr)
        v = tl.load(V_block_ptr)
        k_decay = tl.exp(-s * (BLOCK - (j * CBLOCK + c_array[None, :])))
        kv += tl.dot(k_trans * k_decay, v)
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e
    tl.store(KV_block_ptr, kv)
