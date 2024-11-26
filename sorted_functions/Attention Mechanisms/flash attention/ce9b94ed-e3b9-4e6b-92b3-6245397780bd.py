import triton
import triton.language as tl
import torch

@triton.jit
def _flash_decoding_stage2_kernel(Mid_O, Mid_O_LogExpSum, Ouput,
    mido_batch_stride, mido_heads_stride, mido_partitions_stride,
    mido_dim_stride, mido_les_batch_stride, mido_les_heads_stride,
    mido_les_partitions_stride, o_bs_stride, o_heads_stride, o_dim_stride,
    actual_seq_len, BLOCK_DMODEL: 'tl.constexpr', BLOCK_SEQ: 'tl.constexpr'):
    """Reduction (online softmax)
	"""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_part_v = (batch_idx * mido_batch_stride + head_idx *
        mido_heads_stride + offs_d * mido_dim_stride)
    offs_part_max = (batch_idx * mido_les_batch_stride + head_idx *
        mido_les_heads_stride)
    part_v_ptrs = Mid_O + offs_part_v
    part_max_ptrs = Mid_O_LogExpSum + offs_part_max
    d_i = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    m_i = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    num_partitions = (actual_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    for _ in range(0, num_partitions, 1):
        part_v = tl.load(part_v_ptrs)
        part_max = tl.load(part_max_ptrs)
        m_ij = tl.maximum(part_max, m_i)
        p = tl.exp(part_v - m_ij)
        alpha = tl.exp(m_i - m_ij)
        d_i = d_i * alpha + p
        acc *= alpha
        acc += p * part_v
        m_i = m_ij
        part_v_ptrs += mido_partitions_stride
        part_max_ptrs += mido_les_partitions_stride
    offs_out = (batch_idx * o_bs_stride + head_idx * o_heads_stride + 
        offs_d * o_dim_stride)
    tl.store(Ouput + offs_out, acc / d_i)
