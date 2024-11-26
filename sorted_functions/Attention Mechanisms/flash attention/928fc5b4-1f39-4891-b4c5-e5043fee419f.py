import triton
import triton.language as tl
import torch

@triton.jit
def _flash_decoding_fwd_kernel(Q, KCache, VCache, mid_o, mid_o_lse,
    kv_seq_len, q_len: 'tl.constexpr', batch_size, sm_scale, stride_qt,
    stride_qh, stride_q_qlen, stride_qd, stride_kb, stride_kh, stride_kt,
    stride_kd, stride_vb, stride_vh, stride_vt, stride_vd, stride_mid_ot,
    stride_mid_oh, stride_mid_ob, stride_mid_oqlen, stride_mid_od,
    stride_mid_o_lset, stride_mid_o_lseh, stride_mid_o_lseb, KV_GROUPS:
    'tl.constexpr', BLOCK_KV: 'tl.constexpr', HEAD_DIM: 'tl.constexpr'):
    cur_token_idx = tl.program_id(0)
    cur_head_idx = tl.program_id(1)
    block_start_kv = tl.program_id(2)
    cur_kv_seq_len = tl.load(kv_seq_len + cur_token_idx)
    if block_start_kv * BLOCK_KV >= cur_kv_seq_len:
        return
    offsets_dmodel = tl.arange(0, HEAD_DIM)
    offsets_q = cur_token_idx * stride_qt + cur_head_idx * stride_qh
    Q_block_ptr = tl.make_block_ptr(base=Q + offsets_q, shape=(q_len,
        HEAD_DIM), strides=(stride_q_qlen, stride_qd), offsets=(0, 0),
        block_shape=(q_len, HEAD_DIM), order=(0, 1))
    q = tl.load(Q_block_ptr)
    cur_kv_head_idx = cur_head_idx // KV_GROUPS
    cur_k_offset = (cur_token_idx * stride_kb + cur_kv_head_idx * stride_kh +
        block_start_kv * BLOCK_KV * stride_kt)
    cur_v_offset = (cur_token_idx * stride_vb + cur_kv_head_idx * stride_vh +
        block_start_kv * BLOCK_KV * stride_vt)
    K_block_ptr = tl.make_block_ptr(base=KCache + cur_k_offset, shape=(
        cur_kv_seq_len, HEAD_DIM), strides=(stride_kd, stride_kt), offsets=
        (0, 0), block_shape=(HEAD_DIM, BLOCK_KV), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=VCache + cur_v_offset, shape=(
        cur_kv_seq_len, HEAD_DIM), strides=(stride_vt, stride_vd), offsets=
        (0, 0), block_shape=(BLOCK_KV, HEAD_DIM), order=(0, 1))
    block_mask = block_start_kv * BLOCK_KV + tl.arange(0, BLOCK_KV
        ) < cur_kv_seq_len
    k_cur_block = tl.load(K_block_ptr)
    v_cur_block = tl.load(V_block_ptr)
    acc = tl.zeros([q_len, HEAD_DIM], dtype=tl.float32)
    S_ij = tl.zeros([q_len, BLOCK_KV], dtype=tl.float32)
    S_ij += tl.dot(q, k_cur_block)
    S_ij = tl.where(block_mask[None, :], S_ij, float('-inf'))
    S_ij *= sm_scale
    m = tl.max(S_ij, 1)
    S_ij -= m[:, None]
    p_ij_hat = tl.exp(S_ij)
    l_i = tl.sum(p_ij_hat, 1)
    p_ij_hat = p_ij_hat
    acc += tl.dot(p_ij_hat, v_cur_block)
    acc = acc / l_i[:, None]
    cur_offest_mid = (cur_token_idx * stride_mid_ot + cur_head_idx *
        stride_mid_oh + block_start_kv * stride_mid_ob)
    offsets_mid_o = tl.make_block_ptr(base=mid_o + cur_offest_mid, shape=(
        q_len, HEAD_DIM), strides=(stride_mid_oqlen, stride_mid_od),
        offsets=(0, 0), block_shape=(q_len, HEAD_DIM), order=(0, 1))
    tl.store(offsets_mid_o, acc)
    offsets_qlen = tl.arange(0, q_len)
    offsets_mid_o_lse = (cur_token_idx * stride_mid_o_lset + cur_head_idx *
        stride_mid_o_lseh + block_start_kv * stride_mid_o_lseb + offsets_qlen)
    tl.store(mid_o_lse + offsets_mid_o_lse, m + tl.log(l_i))
