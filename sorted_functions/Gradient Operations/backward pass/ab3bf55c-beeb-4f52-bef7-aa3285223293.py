import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_intra_kernel(Q, K, V, S, DO, DQ, DK, DV, b: 'tl.constexpr', h:
    'tl.constexpr', n: 'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr',
    BLOCK: 'tl.constexpr', NUM_BLOCK: 'tl.constexpr', CBLOCK:
    'tl.constexpr', NUM_CBLOCK: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_h = off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    block_offset = off_block * BLOCK + tl.arange(0, BLOCK)
    Q_trans_block_ptr = Q + qk_offset + block_offset[None, :] * d + tl.arange(
        0, d)[:, None]
    K_block_ptr = K + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[
        None, :]
    V_trans_block_ptr = V + v_offset + block_offset[None, :] * e + tl.arange(
        0, e)[:, None]
    DQ_block_ptr = DQ + qk_offset + block_offset[:, None] * d + tl.arange(0, d
        )[None, :]
    DK_trans_block_ptr = DK + qk_offset + block_offset[None, :
        ] * d + tl.arange(0, d)[:, None]
    DV_block_ptr = DV + v_offset + block_offset[:, None] * e + tl.arange(0, e)[
        None, :]
    DO_block_ptr = DO + o_offset + block_offset[:, None] * e + tl.arange(0, e)[
        None, :]
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)
    array = tl.arange(0, BLOCK)
    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float('-inf'))
    diag_decay = tl.exp(s_index)
    diag_decay_trans = tl.trans(diag_decay)
    k = tl.load(K_block_ptr, mask=block_offset[:, None] < n, other=0.0)
    v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < n,
        other=0.0)
    do = tl.load(DO_block_ptr, mask=block_offset[:, None] < n, other=0.0)
    q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < n,
        other=0.0)
    dqk = tl.dot(do, v_trans) * diag_decay
    dq_intra = tl.dot(dqk, k)
    dk_intra_trans = tl.dot(q_trans, dqk)
    qk_trans = tl.dot(k, q_trans) * diag_decay_trans
    dv_intra = tl.dot(qk_trans, do)
    dq = dq_intra
    dk_trans = dk_intra_trans
    dv = dv_intra
    tl.store(DQ_block_ptr, dq, mask=block_offset[:, None] < n)
    tl.store(DK_trans_block_ptr, dk_trans, mask=block_offset[None, :] < n)
    tl.store(DV_block_ptr, dv, mask=block_offset[:, None] < n)
