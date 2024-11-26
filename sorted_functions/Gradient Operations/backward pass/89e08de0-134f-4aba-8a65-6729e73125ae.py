import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_none_diag_kernel(Q, K, V, S, DO, DQ, DK, DV, DKV, b:
    'tl.constexpr', h: 'tl.constexpr', n: 'tl.constexpr', d: 'tl.constexpr',
    e: 'tl.constexpr', BLOCK: 'tl.constexpr', NUM_BLOCK: 'tl.constexpr',
    CBLOCK: 'tl.constexpr', NUM_CBLOCK: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_h = off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    kv_offset = off_bh * d * e
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)
    block_decay = tl.exp(-s * BLOCK)
    DQ_block_ptr = DQ + qk_offset + qk_block_offset + tl.arange(0, CBLOCK)[
        :, None] * d + tl.arange(0, d)[None, :]
    K_block_ptr = K + qk_offset + qk_block_offset + tl.arange(0, CBLOCK)[:,
        None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = V + v_offset + v_block_offset + tl.arange(0, CBLOCK)[
        None, :] * e + tl.arange(0, e)[:, None]
    DO_block_ptr = DO + o_offset + o_block_offset + tl.arange(0, CBLOCK)[:,
        None] * e + tl.arange(0, e)[None, :]
    DKV_block_ptr = DKV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(
        0, e)[None, :]
    c_array = tl.arange(0, CBLOCK)
    kv_trans = tl.zeros([e, d], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        for j in range(NUM_CBLOCK):
            q_decay = tl.exp(-s * (j * CBLOCK + c_array[:, None]))
            do = tl.load(DO_block_ptr)
            dq_none_diag = tl.dot(do, kv_trans) * q_decay
            dq = dq_none_diag + tl.load(DQ_block_ptr)
            tl.store(DQ_block_ptr, dq)
            DQ_block_ptr += CBLOCK * d
            DO_block_ptr += CBLOCK * e
        kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
        for j in range(NUM_CBLOCK):
            v_trans = tl.load(V_trans_block_ptr)
            k = tl.load(K_block_ptr)
            k_decay = tl.exp(-s * (BLOCK - (j * CBLOCK + c_array[:, None])))
            kv_trans_current += tl.dot(v_trans, k * k_decay)
            K_block_ptr += CBLOCK * d
            V_trans_block_ptr += CBLOCK * e
        kv_trans = block_decay * kv_trans + kv_trans_current
    Q_trans_block_ptr = Q + qk_offset + qk_block_offset + n * d + tl.arange(
        0, CBLOCK)[None, :] * d + tl.arange(0, d)[:, None]
    K_block_ptr = K + qk_offset + qk_block_offset + n * d + tl.arange(0, CBLOCK
        )[:, None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = V + v_offset + v_block_offset + n * e + tl.arange(0,
        CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    DK_trans_block_ptr = DK + qk_offset + qk_block_offset + n * d + tl.arange(
        0, CBLOCK)[None, :] * d + tl.arange(0, d)[:, None]
    DV_block_ptr = DV + v_offset + v_block_offset + n * e + tl.arange(0, CBLOCK
        )[:, None] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + o_block_offset + n * e + tl.arange(0, CBLOCK
        )[:, None] * e + tl.arange(0, e)[None, :]
    dkv = tl.zeros([d, e], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        for j in range(NUM_CBLOCK - 1, -1, -1):
            K_block_ptr -= CBLOCK * d
            V_trans_block_ptr -= CBLOCK * e
            DK_trans_block_ptr -= CBLOCK * d
            DV_block_ptr -= CBLOCK * e
            k = tl.load(K_block_ptr)
            v_trans = tl.load(V_trans_block_ptr)
            k_decay_trans = tl.exp(-s * (BLOCK - (j * CBLOCK + c_array[None,
                :])))
            k_decay = tl.exp(-s * (BLOCK - (j * CBLOCK + c_array[:, None])))
            dk_none_diag_trans = tl.dot(dkv, v_trans) * k_decay_trans
            dv_none_diag = tl.dot(k, dkv) * k_decay
            dk_trans = dk_none_diag_trans + tl.load(DK_trans_block_ptr)
            dv = dv_none_diag + tl.load(DV_block_ptr)
            tl.store(DK_trans_block_ptr, dk_trans)
            tl.store(DV_block_ptr, dv)
        dkv_current = tl.zeros([d, e], dtype=tl.float32)
        for j in range(NUM_CBLOCK - 1, -1, -1):
            DO_block_ptr -= CBLOCK * e
            Q_trans_block_ptr -= CBLOCK * d
            do = tl.load(DO_block_ptr)
            q_trans = tl.load(Q_trans_block_ptr)
            q_decay_trans = tl.exp(-s * (j * CBLOCK + c_array[None, :]))
            dkv_current += tl.dot(q_trans * q_decay_trans, do)
        dkv = block_decay * dkv + dkv_current
    tl.store(DKV_block_ptr, dkv)
