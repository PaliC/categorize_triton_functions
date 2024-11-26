import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_none_diag_kernel(Q, K, V, Out, S, KV, GKV, b: 'tl.constexpr', h:
    'tl.constexpr', n: 'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr',
    BLOCK: 'tl.constexpr', NUM_BLOCK: 'tl.constexpr', D_FBLOCK:
    'tl.constexpr', E_FBLOCK: 'tl.constexpr', NUM_FBLOCK: 'tl.constexpr',
    CBLOCK: 'tl.constexpr', NUM_CBLOCK: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    off_e = tl.program_id(2)
    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK
    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e + e_offset
    gkv_offset = off_bh * d * e + e_offset
    Q_block_ptr = Q + q_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(
        0, d)[None, :]
    O_block_ptr = Out + o_offset + tl.arange(0, CBLOCK)[:, None
        ] * e + tl.arange(0, E_FBLOCK)[None, :]
    KV_block_ptr = KV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(
        0, E_FBLOCK)[None, :]
    GKV_block_ptr = GKV + gkv_offset + tl.arange(0, d)[:, None
        ] * e + tl.arange(0, E_FBLOCK)[None, :]
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)
    c_array = tl.arange(0, CBLOCK)
    GKV = tl.load(GKV_block_ptr)
    kv = tl.load(KV_block_ptr)
    q = tl.load(Q_block_ptr)
    q_decay = tl.exp(-s * (c_offset + c_array[:, None]))
    qkv_none_diag = tl.dot(q, kv) * q_decay + tl.dot(q, GKV) * tl.exp(-s *
        (c_offset + c_array[:, None] + n_offset))
    qkv_diag = tl.load(O_block_ptr)
    qkv = qkv_diag + qkv_none_diag
    tl.store(O_block_ptr, qkv)
