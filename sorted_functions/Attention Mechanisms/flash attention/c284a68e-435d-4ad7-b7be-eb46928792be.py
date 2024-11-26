import triton
import triton.language as tl
import torch

@triton.autotune([triton.Config({'BLOCK_SIZE_Q': BLOCK_SIZE_Q,
    'BLOCK_SIZE_KV': BLOCK_SIZE_KV}, num_stages=num_stages, num_warps=
    num_warps) for BLOCK_SIZE_Q in [64, 128] for BLOCK_SIZE_KV in [32, 64] for
    num_stages in [3, 4, 7] for num_warps in [2, 4]], key=['SEQ_LEN',
    'HEAD_DIM'])
@triton.jit
def _attn_fwd(Q, K, V, softmax_scale, M, O, stride_Q_batch, stride_Q_head,
    stride_Q_seq, stride_Q_dim, stride_K_batch, stride_K_head, stride_K_seq,
    stride_K_dim, stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim, BATCH_SIZE,
    NUM_HEADS: 'tl.constexpr', SEQ_LEN: 'tl.constexpr', HEAD_DIM:
    'tl.constexpr', BLOCK_SIZE_Q: 'tl.constexpr', BLOCK_SIZE_KV:
    'tl.constexpr', STAGE: 'tl.constexpr'):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    qvk_offset = index_batch * stride_Q_batch + index_head * stride_Q_head
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(SEQ_LEN,
        HEAD_DIM), strides=(stride_Q_seq, stride_Q_dim), offsets=(
        block_index_q * BLOCK_SIZE_Q, 0), block_shape=(BLOCK_SIZE_Q,
        HEAD_DIM), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=V + qvk_offset, shape=(SEQ_LEN,
        HEAD_DIM), strides=(stride_V_seq, stride_V_dim), offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qvk_offset, shape=(HEAD_DIM,
        SEQ_LEN), strides=(stride_K_dim, stride_K_seq), offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(base=O + qvk_offset, shape=(SEQ_LEN,
        HEAD_DIM), strides=(stride_O_seq, stride_O_dim), offsets=(
        block_index_q * BLOCK_SIZE_Q, 0), block_shape=(BLOCK_SIZE_Q,
        HEAD_DIM), order=(1, 0))
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    Q_block = tl.load(Q_block_ptr)
    if STAGE == 1 or STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(O_block, l_i, m_i, Q_block,
            K_block_ptr, V_block_ptr, block_index_q, softmax_scale,
            BLOCK_SIZE_Q, BLOCK_SIZE_KV, 4 - STAGE, offs_q, offs_kv, SEQ_LEN)
    if STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(O_block, l_i, m_i, Q_block,
            K_block_ptr, V_block_ptr, block_index_q, softmax_scale,
            BLOCK_SIZE_Q, BLOCK_SIZE_KV, 2, offs_q, offs_kv, SEQ_LEN)
    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block)
