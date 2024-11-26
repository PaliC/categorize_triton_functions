import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(weight, input_ids, out, vob_start_id, vob_end_id,
    stride_weight_seq, stride_out_seq, n_ctx, hiden_size: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_NN:
    'tl.constexpr'):
    start_n = tl.program_id(0) * BLOCK_N
    offs_nn = start_n + tl.arange(0, BLOCK_NN)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    for start_nn in range(0, BLOCK_N, BLOCK_NN):
        start_nn = tl.multiple_of(start_nn, BLOCK_NN)
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < n_ctx
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=
            vob_end_id)
        id_mask = (token_ids >= vob_start_id) & (token_ids < vob_end_id)
        token_ids = token_ids - vob_start_id
        dim_mask = offs_d < hiden_size
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(weight + token_ids[:, None] * stride_weight_seq +
            offs_d[None, :], mask=load_mask, other=0.0)
        tl.store(out + offs_seq[:, None] * stride_out_seq + offs_d[None, :],
            vecs, mask=store_mask)
