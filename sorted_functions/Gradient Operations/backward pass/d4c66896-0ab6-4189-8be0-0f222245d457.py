import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kernel(Q, K, V, S, DO, DQ, DK, DV, KV, DKV, b: 'tl.constexpr', h:
    'tl.constexpr', n: 'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr',
    BLOCK: 'tl.constexpr', NUM_BLOCK: 'tl.constexpr', DBLOCK:
    'tl.constexpr', NUM_DBLOCK: 'tl.constexpr', EBLOCK: 'tl.constexpr',
    NUM_EBLOCK: 'tl.constexpr'):
    off_d = tl.program_id(0)
    off_e = tl.program_id(1)
    off_bh = tl.program_id(2)
    off_h = off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    kv_offset = off_bh * d * e
    d_offset = off_d * DBLOCK
    e_offset = off_e * EBLOCK
    dqk_offset = off_e * b * h * n * d
    dv_offset = off_d * b * h * n * e
    d_offset = off_d * DBLOCK
    e_offset = off_e * EBLOCK
    kv_d_offset = d_offset * e
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)
    block_decay = tl.exp(-s * BLOCK)
    DQ_block_ptr = DQ + qk_offset + dqk_offset + d_offset + tl.arange(0, BLOCK
        )[:, None] * d + tl.arange(0, DBLOCK)[None, :]
    K_block_ptr = K + qk_offset + d_offset + tl.arange(0, BLOCK)[:, None
        ] * d + tl.arange(0, DBLOCK)[None, :]
    V_trans_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK)[None, :
        ] * e + tl.arange(0, EBLOCK)[:, None]
    DO_block_ptr = DO + o_offset + e_offset + tl.arange(0, BLOCK)[:, None
        ] * e + tl.arange(0, EBLOCK)[None, :]
    KV_trans_block_ptr = KV + kv_offset + kv_d_offset + e_offset + tl.arange(
        0, DBLOCK)[None, :] * e + tl.arange(0, EBLOCK)[:, None]
    DKV_block_ptr = DKV + kv_offset + kv_d_offset + e_offset + tl.arange(0,
        DBLOCK)[:, None] * e + tl.arange(0, EBLOCK)[None, :]
    array = tl.arange(0, BLOCK)
    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float('-inf'))
    diag_decay = tl.exp(s_index)
    diag_decay_trans = tl.trans(diag_decay)
    KV_trans = tl.load(KV_trans_block_ptr)
    kv_trans = tl.zeros([EBLOCK, DBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        q_decay = tl.exp(-s * array[:, None])
        k_decay = tl.exp(-s * (BLOCK - array[:, None]))
        do = tl.load(DO_block_ptr)
        k = tl.load(K_block_ptr)
        v_trans = tl.load(V_trans_block_ptr)
        dq_none_diag = tl.dot(do, kv_trans) * q_decay + tl.dot(do, KV_trans
            ) * tl.exp(-s * (i * BLOCK + array[:, None]))
        dqk = tl.dot(do, v_trans) * diag_decay
        dq_diag = tl.dot(dqk, k)
        dq = dq_none_diag + dq_diag
        tl.store(DQ_block_ptr, dq)
        DQ_block_ptr += BLOCK * d
        DO_block_ptr += BLOCK * e
        K_block_ptr += BLOCK * d
        V_trans_block_ptr += BLOCK * e
        kv_trans = block_decay * kv_trans + tl.dot(v_trans, k * k_decay)
    Q_trans_block_ptr = Q + qk_offset + d_offset + n * d + tl.arange(0, BLOCK)[
        None, :] * d + tl.arange(0, DBLOCK)[:, None]
    K_block_ptr = K + qk_offset + d_offset + n * d + tl.arange(0, BLOCK)[:,
        None] * d + tl.arange(0, DBLOCK)[None, :]
    V_trans_block_ptr = V + v_offset + e_offset + n * e + tl.arange(0, BLOCK)[
        None, :] * e + tl.arange(0, EBLOCK)[:, None]
    DK_trans_block_ptr = (DK + qk_offset + dqk_offset + d_offset + n * d + 
        tl.arange(0, BLOCK)[None, :] * d + tl.arange(0, DBLOCK)[:, None])
    DV_block_ptr = DV + v_offset + dv_offset + e_offset + n * e + tl.arange(
        0, BLOCK)[:, None] * e + tl.arange(0, EBLOCK)[None, :]
    DO_block_ptr = DO + o_offset + e_offset + n * e + tl.arange(0, BLOCK)[:,
        None] * e + tl.arange(0, EBLOCK)[None, :]
    DKV = tl.load(DKV_block_ptr)
    dkv = tl.zeros([DBLOCK, EBLOCK], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        K_block_ptr -= BLOCK * d
        V_trans_block_ptr -= BLOCK * e
        DK_trans_block_ptr -= BLOCK * d
        DV_block_ptr -= BLOCK * e
        DO_block_ptr -= BLOCK * e
        Q_trans_block_ptr -= BLOCK * d
        k = tl.load(K_block_ptr)
        v_trans = tl.load(V_trans_block_ptr)
        do = tl.load(DO_block_ptr)
        q_trans = tl.load(Q_trans_block_ptr)
        k_decay_trans = tl.exp(-s * (BLOCK - array[None, :]))
        k_decay = tl.exp(-s * (BLOCK - array[:, None]))
        q_decay_trans = tl.exp(-s * array[None, :])
        dqk = tl.dot(do, v_trans) * diag_decay
        dk_diag_trans = tl.dot(q_trans, dqk)
        dk_none_diag_trans = tl.dot(dkv, v_trans) * k_decay_trans + tl.dot(DKV,
            v_trans) * tl.exp(-s * (n - i * BLOCK - array[None, :]))
        dk_trans = dk_none_diag_trans + dk_diag_trans
        qk_trans = tl.dot(k, q_trans) * diag_decay_trans
        dv_diag = tl.dot(qk_trans, do)
        dv_none_diag = tl.dot(k, dkv) * k_decay + tl.dot(k, DKV) * tl.exp(-
            s * (n - i * BLOCK - array[:, None]))
        dv = dv_none_diag + dv_diag
        tl.store(DK_trans_block_ptr, dk_trans)
        tl.store(DV_block_ptr, dv)
        dkv = block_decay * dkv + tl.dot(q_trans * q_decay_trans, do)
    DKV = tl.exp(-s * n) * DKV + dkv
    tl.store(DKV_block_ptr, DKV)
