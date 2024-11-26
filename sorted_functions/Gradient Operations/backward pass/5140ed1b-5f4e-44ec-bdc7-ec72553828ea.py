import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_diag_kernel(Q, K, V, S, DO, DQ, DK, DV, b: 'tl.constexpr', h:
    'tl.constexpr', n: 'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr',
    BLOCK: 'tl.constexpr', NUM_BLOCK: 'tl.constexpr', CBLOCK:
    'tl.constexpr', NUM_CBLOCK: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_h = off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e
    Q_trans_block_ptr = Q + qk_offset + qk_block_offset + tl.arange(0, BLOCK)[
        None, :] * d + tl.arange(0, d)[:, None]
    K_block_ptr = K + qk_offset + qk_block_offset + tl.arange(0, BLOCK)[:, None
        ] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = V + v_offset + v_block_offset + tl.arange(0, BLOCK)[
        None, :] * e + tl.arange(0, e)[:, None]
    DQ_block_ptr = DQ + qk_offset + qk_block_offset + tl.arange(0, BLOCK)[:,
        None] * d + tl.arange(0, d)[None, :]
    DK_trans_block_ptr = DK + qk_offset + qk_block_offset + tl.arange(0, BLOCK
        )[None, :] * d + tl.arange(0, d)[:, None]
    DV_block_ptr = DV + v_offset + v_block_offset + tl.arange(0, BLOCK)[:, None
        ] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + o_block_offset + tl.arange(0, BLOCK)[:, None
        ] * e + tl.arange(0, e)[None, :]
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)
    array = tl.arange(0, BLOCK)
    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float('-inf'))
    diag_decay = tl.exp(s_index)
    diag_decay_trans = tl.trans(diag_decay)
    k = tl.load(K_block_ptr)
    v_trans = tl.load(V_trans_block_ptr)
    do = tl.load(DO_block_ptr)
    q_trans = tl.load(Q_trans_block_ptr)
    dqk = tl.dot(do, v_trans) * diag_decay
    dq_diag = tl.dot(dqk, k)
    dq = dq_diag
    dk_diag_trans = tl.dot(q_trans, dqk)
    qk_trans = tl.dot(k, q_trans) * diag_decay_trans
    dv_diag = tl.dot(qk_trans, do)
    dk_trans = dk_diag_trans
    dv = dv_diag
    tl.store(DQ_block_ptr, dq)
    tl.store(DK_trans_block_ptr, dk_trans)
    tl.store(DV_block_ptr, dv)
