import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_dkv_parallel(Q, DO, S, DKV, b: 'tl.constexpr', h: 'tl.constexpr',
    n: 'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr', BLOCK:
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
    qk_block_offset = block_offset * d
    o_block_offset = block_offset * e
    kv_block_offset = off_block * d * e
    qk_offset = off_bh * n * d
    o_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK
    DKV_block_ptr = (DKV + kv_offset + kv_block_offset + d_offset * e +
        e_offset + tl.arange(0, D_FBLOCK)[:, None] * e + tl.arange(0,
        E_FBLOCK)[None, :])
    Q_trans_block_ptr = Q + qk_offset + qk_block_offset + d_offset + tl.arange(
        0, CBLOCK)[None, :] * d + tl.arange(0, D_FBLOCK)[:, None]
    DO_block_ptr = DO + o_offset + o_block_offset + e_offset + tl.arange(0,
        CBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    s_ptrs = S + off_h
    s = tl.load(s_ptrs)
    c_array = tl.arange(0, CBLOCK)
    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for j in range(NUM_CBLOCK):
        do = tl.load(DO_block_ptr)
        q_trans = tl.load(Q_trans_block_ptr)
        q_decay_trans = tl.exp(-s * (j * CBLOCK + c_array[None, :]))
        dkv += tl.dot(q_trans * q_decay_trans, do)
        DO_block_ptr += CBLOCK * e
        Q_trans_block_ptr += CBLOCK * d
    tl.store(DKV_block_ptr, dkv)
