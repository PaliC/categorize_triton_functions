import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_inter_kernel(Q, K, V, S, DO, DQ, DK, DV, b: 'tl.constexpr', h:
    'tl.constexpr', n: 'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr',
    BLOCK: 'tl.constexpr', NUM_BLOCK: 'tl.constexpr', CBLOCK:
    'tl.constexpr', NUM_CBLOCK: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    S_block_ptr = S + off_h
    DQ_block_ptr = DQ + qk_offset + tl.arange(0, CBLOCK)[:, None
        ] * d + tl.arange(0, d)[None, :]
    K_block_ptr = K + qk_offset + tl.arange(0, CBLOCK)[:, None
        ] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = V + v_offset + tl.arange(0, CBLOCK)[None, :
        ] * e + tl.arange(0, e)[:, None]
    DO_block_ptr = DO + o_offset + tl.arange(0, CBLOCK)[:, None
        ] * e + tl.arange(0, e)[None, :]
    off_block1 = tl.arange(0, CBLOCK)
    off_block2 = tl.arange(0, CBLOCK)
    c_array = tl.arange(0, CBLOCK)
    s = tl.load(S_block_ptr)
    block_decay = tl.exp(-s * BLOCK)
    kv_trans = tl.zeros([e, d], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        for j in range(NUM_CBLOCK):
            if i > 0:
                q_decay = tl.exp(-s * (j * CBLOCK + c_array[:, None]))
                do = tl.load(DO_block_ptr, mask=off_block1[:, None] < n,
                    other=0.0)
                dq_inter = tl.dot(do, kv_trans) * q_decay
                dq = dq_inter + tl.load(DQ_block_ptr, mask=off_block1[:,
                    None] < n, other=0.0)
                tl.store(DQ_block_ptr, dq, mask=off_block1[:, None] < n)
            DQ_block_ptr += CBLOCK * d
            DO_block_ptr += CBLOCK * e
            off_block1 += CBLOCK
        kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
        for j in range(NUM_CBLOCK):
            v_trans = tl.load(V_trans_block_ptr, mask=off_block2[None, :] <
                n, other=0.0)
            k = tl.load(K_block_ptr, mask=off_block2[:, None] < n, other=0.0)
            k_decay = tl.exp(-s * (BLOCK - (j * CBLOCK + c_array[:, None])))
            kv_trans_current += tl.dot(v_trans, k * k_decay)
            K_block_ptr += CBLOCK * d
            V_trans_block_ptr += CBLOCK * e
            off_block2 += CBLOCK
        kv_trans = block_decay * kv_trans + kv_trans_current
    m = NUM_BLOCK * BLOCK
    off_block1 = m + tl.arange(0, CBLOCK)
    off_block2 = m + tl.arange(0, CBLOCK)
    Q_trans_block_ptr = Q + qk_offset + m * d + tl.arange(0, CBLOCK)[None, :
        ] * d + tl.arange(0, d)[:, None]
    K_block_ptr = K + qk_offset + m * d + tl.arange(0, CBLOCK)[:, None
        ] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = V + v_offset + m * e + tl.arange(0, CBLOCK)[None, :
        ] * e + tl.arange(0, e)[:, None]
    DK_trans_block_ptr = DK + qk_offset + m * d + tl.arange(0, CBLOCK)[None, :
        ] * d + tl.arange(0, d)[:, None]
    DV_block_ptr = DV + v_offset + m * e + tl.arange(0, CBLOCK)[:, None
        ] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + m * e + tl.arange(0, CBLOCK)[:, None
        ] * e + tl.arange(0, e)[None, :]
    dkv = tl.zeros([d, e], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        for j in range(NUM_CBLOCK - 1, -1, -1):
            K_block_ptr -= CBLOCK * d
            V_trans_block_ptr -= CBLOCK * e
            DK_trans_block_ptr -= CBLOCK * d
            DV_block_ptr -= CBLOCK * e
            off_block1 -= CBLOCK
            if i < NUM_BLOCK - 1:
                k = tl.load(K_block_ptr, mask=off_block1[:, None] < n,
                    other=0.0)
                v_trans = tl.load(V_trans_block_ptr, mask=off_block1[None,
                    :] < n, other=0.0)
                k_decay_trans = tl.exp(-s * (BLOCK - (j * CBLOCK + c_array[
                    None, :])))
                k_decay = tl.exp(-s * (BLOCK - (j * CBLOCK + c_array[:, None]))
                    )
                dk_inter_trans = tl.dot(dkv, v_trans) * k_decay_trans
                dv_inter = tl.dot(k, dkv) * k_decay
                dk_trans = dk_inter_trans + tl.load(DK_trans_block_ptr,
                    mask=off_block1[None, :] < n, other=0.0)
                dv = dv_inter + tl.load(DV_block_ptr, mask=off_block1[:,
                    None] < n, other=0.0)
                tl.store(DK_trans_block_ptr, dk_trans, mask=off_block1[None,
                    :] < n)
                tl.store(DV_block_ptr, dv, mask=off_block1[:, None] < n)
        dkv_current = tl.zeros([d, e], dtype=tl.float32)
        for j in range(NUM_CBLOCK - 1, -1, -1):
            DO_block_ptr -= CBLOCK * e
            Q_trans_block_ptr -= CBLOCK * d
            off_block2 -= CBLOCK
            do = tl.load(DO_block_ptr, mask=off_block2[:, None] < n, other=0.0)
            q_trans = tl.load(Q_trans_block_ptr, mask=off_block2[None, :] <
                n, other=0.0)
            q_decay_trans = tl.exp(-s * (j * CBLOCK + c_array[None, :]))
            dkv_current += tl.dot(q_trans * q_decay_trans, do)
        dkv = block_decay * dkv + dkv_current
