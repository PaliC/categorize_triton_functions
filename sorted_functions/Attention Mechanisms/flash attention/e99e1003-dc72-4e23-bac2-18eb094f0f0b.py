import triton
import triton.language as tl
import torch

@triton.jit
def _flash_decoding_fwd_reduce_kernel(mid_o, mid_o_lse, O, kv_seq_len,
    q_len: 'tl.constexpr', batch_size, stride_mid_ot, stride_mid_oh,
    stride_mid_ob, stride_mid_oqlen, stride_mid_od, stride_o_lset,
    stride_o_lseh, stride_o_lseb, stride_o_lseqlen, stride_ot, stride_oh,
    stride_oqlen, BLOCK_KV: 'tl.constexpr', HEAD_DIM: 'tl.constexpr'):
    cur_token_idx = tl.program_id(0)
    cur_head_idx = tl.program_id(1)
    cur_q_idx = tl.program_id(2)
    cur_kv_seq_len = tl.load(kv_seq_len + cur_token_idx)
    offsets_dmodel = tl.arange(0, HEAD_DIM)
    kv_split_num = (cur_kv_seq_len + BLOCK_KV - 1) // BLOCK_KV
    m_i = float('-inf')
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    offsets_mid_o = (cur_token_idx * stride_mid_ot + cur_head_idx *
        stride_mid_oh + cur_q_idx * stride_mid_oqlen + offsets_dmodel)
    offset_mid_lse = (cur_token_idx * stride_o_lset + cur_head_idx *
        stride_o_lseh + cur_q_idx * stride_o_lseqlen)
    for block_i in range(0, kv_split_num, 1):
        mid_o_block = tl.load(mid_o + offsets_mid_o + block_i * stride_mid_ob)
        lse = tl.load(mid_o_lse + offset_mid_lse + block_i * stride_o_lseb)
        m_ij = tl.maximum(m_i, lse)
        scale = tl.exp(m_i - m_ij)
        acc = acc * scale
        lse -= m_ij
        exp_logic = tl.exp(lse)
        acc += exp_logic * mid_o_block
        l_i = scale * l_i + exp_logic
        m_i = m_ij
    acc = acc / l_i
    offsets_O = (cur_token_idx * stride_ot + cur_head_idx * stride_oh + 
        cur_q_idx * stride_oqlen + offsets_dmodel)
    tl.store(O + offsets_O, acc)
    return l_i
