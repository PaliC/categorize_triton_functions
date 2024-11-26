import triton
import triton.language as tl
import torch

@triton.jit
def _attn_bwd_preprocess(O, dO, D, SEQ_LEN, BLOCK_SIZE_Q: 'tl.constexpr',
    HEAD_DIM: 'tl.constexpr'):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)
    O_block = tl.load(O + index_batch_head * HEAD_DIM * SEQ_LEN + offs_q[:,
        None] * HEAD_DIM + offs_dim[None, :])
    dO_block = tl.load(dO + index_batch_head * HEAD_DIM * SEQ_LEN + offs_q[
        :, None] * HEAD_DIM + offs_dim[None, :])
    D_block = tl.sum(dO_block * O_block, axis=1)
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)
