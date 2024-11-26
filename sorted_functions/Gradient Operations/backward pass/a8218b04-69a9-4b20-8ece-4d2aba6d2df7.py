import triton
import triton.language as tl
import torch

@triton.jit
def _attn_bwd_dq(Q, K, V, softmax_scale, dO, dQ, dK, dV, M, D, stride_batch,
    stride_head, stride_seq, stride_dim, NUM_HEADS, SEQ_LEN, BLOCK_Q:
    'tl.constexpr', BLOCK_KV: 'tl.constexpr', HEAD_DIM: 'tl.constexpr',
    STAGE: 'tl.constexpr'):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = stride_batch * index_batch + stride_head * index_head
    offset_batch_head_seq = index_batch_head * SEQ_LEN
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head
    M += offset_batch_head_seq
    D += offset_batch_head_seq
    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] *
        stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :
        ] * stride_dim)
    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]
    offs_kv = tl.arange(0, BLOCK_KV)
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None
        ] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None
        ] * stride_dim
    Di = tl.load(D + offs_q)
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)
        if STAGE == 3:
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)
        dP_block = tl.dot(dO_block, V_T_block)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq
    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :
        ] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)
