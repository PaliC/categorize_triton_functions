import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_dkv_reduce(Q, DO, S, DKV, b: 'tl.constexpr', h: 'tl.constexpr', n:
    'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr', BLOCK:
    'tl.constexpr', NUM_BLOCK: 'tl.constexpr', D_FBLOCK: 'tl.constexpr',
    E_FBLOCK: 'tl.constexpr', NUM_FBLOCK: 'tl.constexpr', CBLOCK:
    'tl.constexpr', NUM_CBLOCK: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK
    DKV_block_ptr = (DKV + kv_offset + d_offset * e + e_offset + NUM_BLOCK *
        d * e + tl.arange(0, D_FBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK
        )[None, :])
    s_ptrs = S + off_h
    s = tl.load(s_ptrs)
    block_decay = tl.exp(-s * BLOCK)
    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        DKV_block_ptr -= d * e
        dkv_current = tl.load(DKV_block_ptr)
        tl.store(DKV_block_ptr, dkv)
        dkv = block_decay * dkv + dkv_current
    DKV_block_ptr += NUM_BLOCK * d * e
    tl.store(DKV_block_ptr, dkv)
