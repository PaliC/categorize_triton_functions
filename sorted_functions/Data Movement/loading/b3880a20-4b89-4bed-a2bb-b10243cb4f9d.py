import triton
import triton.language as tl
import torch

@triton.autotune(configs=_config_grouping(), key=['K'])
@triton.heuristics({'NO_K_MASK': lambda args: args['K'] % args['BLOCK_K'] == 0}
    )
@triton.jit
def _group(src_ptr, stride_sn, stride_sk, has_coeff: 'tl.constexpr',
    coeff_ptr, FAN_OUT: 'tl.constexpr', tgt_ptr, stride_tn, stride_ti,
    grouped_idx_ptr, N, K: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K:
    'tl.constexpr', NO_K_MASK: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    N_block_id = pid
    N_blk = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_blk < N
    N_blk = tl.max_contiguous(tl.multiple_of(N_blk % N, BLOCK_N), BLOCK_N)
    N_idx = tl.load(grouped_idx_ptr + N_blk, mask=N_mask, other=0)
    K_blk = tl.arange(0, BLOCK_K)
    src_blk_ptrs = src_ptr + (N_idx // FAN_OUT)[:, None] * stride_sn + K_blk[
        None, :] * stride_sk
    tgt_blk_ptrs = tgt_ptr + N_blk[:, None] * stride_tn + K_blk[None, :
        ] * stride_ti
    if has_coeff:
        c = tl.load(coeff_ptr + N_idx, mask=N_mask)[:, None]
    iters = tl.cdiv(K, BLOCK_K)
    for i in range(0, iters):
        if NO_K_MASK or i < iters - 1:
            block = tl.load(src_blk_ptrs, mask=N_mask[:, None])
            if has_coeff:
                block *= c
            tl.store(tgt_blk_ptrs, block, mask=N_mask[:, None])
        else:
            K_mask = i * BLOCK_K + K_blk < K
            mask = N_mask[:, None] & K_mask[None, :]
            block = tl.load(src_blk_ptrs, mask=mask)
            if has_coeff:
                block *= c
            tl.store(tgt_blk_ptrs, block, mask=mask)
        src_blk_ptrs += BLOCK_K * stride_sk
        tgt_blk_ptrs += BLOCK_K * stride_ti
