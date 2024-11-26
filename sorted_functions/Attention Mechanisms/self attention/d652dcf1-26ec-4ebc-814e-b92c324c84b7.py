import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q, K, V, Out, b: 'tl.constexpr', h: 'tl.constexpr', n:
    'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr', BLOCK:
    'tl.constexpr', NUM_BLOCK: 'tl.constexpr', BLOCK_MODEL: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    e_offset = off_e * BLOCK_MODEL
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :
        ]
    off_block = tl.arange(0, BLOCK)
    index = off_block[:, None] - off_block[None, :]
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        q = tl.load(Q_block_ptr + off_block[:, None] * d, mask=off_block[:,
            None] < n, other=0.0)
        k_trans = tl.load(K_trans_block_ptr + off_block[None, :] * d, mask=
            off_block[None, :] < n, other=0.0)
        v = tl.load(V_block_ptr + off_block[:, None] * e, mask=off_block[:,
            None] < n, other=0.0)
        qk = tl.dot(q, k_trans)
        qk = tl.where(index >= 0, qk, 0)
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv)
        o = o_intra + o_inter
        tl.store(O_block_ptr + off_block[:, None] * e, o, mask=off_block[:,
            None] < n)
        kv += tl.dot(k_trans, v)
        off_block += BLOCK
