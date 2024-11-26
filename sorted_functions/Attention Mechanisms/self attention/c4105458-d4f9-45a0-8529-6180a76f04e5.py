import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q, K, V, O, stride_q_bs, stride_q_head, stride_q_seqlen,
    stride_q_dim, stride_k_bs, stride_k_head, stride_k_seqlen, stride_k_dim,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_dim, stride_o_bs,
    stride_o_head, stride_o_seqlen, stride_o_dim, BS, HEAD, SEQLEN,
    sm_scale, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', DIM:
    'tl.constexpr'):
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)
    qkv_base_offset = off_bs_head * stride_q_head
    Q_block_ptr = tl.make_block_ptr(base=Q + qkv_base_offset, shape=(SEQLEN,
        DIM), strides=(stride_q_seqlen, stride_q_dim), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, DIM), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qkv_base_offset, shape=(DIM,
        SEQLEN), strides=(stride_k_dim, stride_k_seqlen), offsets=(0, 0),
        block_shape=(DIM, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qkv_base_offset, shape=(SEQLEN,
        DIM), strides=(stride_k_seqlen, stride_v_dim), offsets=(0, 0),
        block_shape=(BLOCK_N, DIM), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    for start_n in range(0, SEQLEN, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
        v = v * sm_scale
        kv = tl.dot(k, v, allow_tf32=False)
        out_buffer += tl.dot(q, kv, allow_tf32=False)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    O_block_ptr = tl.make_block_ptr(base=O + qkv_base_offset, shape=(SEQLEN,
        DIM), strides=(stride_o_seqlen, stride_o_dim), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, DIM), order=(1, 0))
    tl.store(O_block_ptr, out_buffer, boundary_check=(0, 1))
