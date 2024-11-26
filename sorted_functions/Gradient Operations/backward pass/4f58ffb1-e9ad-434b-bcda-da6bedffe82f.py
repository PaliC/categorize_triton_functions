import triton
import triton.language as tl
import torch

@triton.jit
def _attn_bwd_dk_dv(Q, K, V, softmax_scale, dO, dQ, dK, dV, M, D,
    stride_batch, stride_head, stride_seq, stride_dim, NUM_HEADS, SEQ_LEN,
    BLOCK_Q: 'tl.constexpr', BLOCK_KV: 'tl.constexpr', HEAD_DIM:
    'tl.constexpr', STAGE: 'tl.constexpr'):
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
    start_kv = index_block_kv * BLOCK_KV
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    K_block = tl.load(K + offs_kv[:, None] * stride_seq + offs_dim[None, :] *
        stride_dim)
    V_block = tl.load(V + offs_kv[:, None] * stride_seq + offs_dim[None, :] *
        stride_dim)
    offs_q = tl.arange(0, BLOCK_Q)
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :
        ] * stride_dim
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        qT_block = tl.load(qT_ptrs)
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])
        if STAGE == 3:
            mask_block = offs_q[None, :] >= offs_kv[:, None]
            P_T_block = tl.where(mask_block, P_T_block, 0.0)
        dO_block = tl.load(dO_ptrs)
        dV_block += tl.dot(P_T_block, dO_block)
        Di = tl.load(D + offs_q)
        dpT_block = tl.dot(V_block, tl.trans(dO_block))
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :
        ] * stride_dim
    tl.store(dV_block_ptrs, dV_block)
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :
        ] * stride_dim
    tl.store(dK_block_ptrs, dK_block)
