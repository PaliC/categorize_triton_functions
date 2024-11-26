import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_diag_kernel(Q, K, V, Out, S, b: 'tl.constexpr', h: 'tl.constexpr',
    n: 'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr', BLOCK:
    'tl.constexpr', NUM_BLOCK: 'tl.constexpr', CBLOCK: 'tl.constexpr',
    NUM_CBLOCK: 'tl.constexpr'):
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK
    off_block = off % NUM_BLOCK
    off_cblock = tl.program_id(1)
    off_h = off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e
    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e
    Q_block_ptr = (Q + qk_offset + qk_block_offset + q_cblock_offset + tl.
        arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :])
    K_trans_block_ptr = K + qk_offset + qk_block_offset + tl.arange(0, CBLOCK)[
        None, :] * d + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + v_block_offset + tl.arange(0, CBLOCK)[:, None
        ] * e + tl.arange(0, e)[None, :]
    O_block_ptr = (Out + o_offset + o_block_offset + o_cblock_offset + tl.
        arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :])
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)
    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK
    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.0)
    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)
    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        s_index = tl.where(diff >= 0, -s_index, float('-inf'))
        decay = tl.exp(s_index)
        k_trans = tl.load(K_trans_block_ptr, mask=kv_index[None, :] < n,
            other=0.0)
        v = tl.load(V_block_ptr, mask=kv_index[:, None] < n, other=0.0)
        qk = tl.dot(q, k_trans) * decay
        qkv += tl.dot(qk, v)
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e
    tl.store(O_block_ptr, qkv, mask=q_index[:, None] < n)
