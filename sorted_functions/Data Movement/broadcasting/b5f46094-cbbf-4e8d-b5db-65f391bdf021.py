import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kv_reduce(K, V, S, KV, b: 'tl.constexpr', h: 'tl.constexpr', n:
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
    KV_block_ptr = KV + kv_offset + d_offset * e + e_offset + tl.arange(0,
        D_FBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    s_ptrs = S + off_h
    s = tl.load(s_ptrs)
    block_decay = tl.exp(-s * BLOCK)
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        kv_current = tl.load(KV_block_ptr)
        tl.store(KV_block_ptr, kv)
        kv = block_decay * kv + kv_current
        KV_block_ptr += d * e
    tl.store(KV_block_ptr, kv)
