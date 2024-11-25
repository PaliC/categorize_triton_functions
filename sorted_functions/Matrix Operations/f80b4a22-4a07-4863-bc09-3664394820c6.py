import triton
import triton.language as tl
import torch

@triton.autotune(configs=_scatter2scatter_configs(), key=['M', 'N', 'K'])
@triton.heuristics({'NO_K_MASK': lambda args: args['K'] % args['BLOCK_K'] ==
    0, 'NO_N_MASK': lambda args: args['N'] % args['BLOCK_N'] == 0})
@triton.jit
def _scatter2scatter(X_ptr, stride_xm, stride_xk, W_ptr, stride_we,
    stride_wk, stride_wn, Y_ptr, stride_ym, stride_yn, grouped_idx_ptr,
    expert_idxs_ptr, block_start_idx_ptr, FAN_OUT: 'tl.constexpr', M, K:
    'tl.constexpr', N: 'tl.constexpr', E: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr',
    ACC_TYPE: 'tl.constexpr', OUT_M, allow_tf32: 'tl.constexpr', x_grouped:
    'tl.constexpr', y_grouped: 'tl.constexpr', NO_K_MASK: 'tl.constexpr',
    NO_N_MASK: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT
    M_range = tl.arange(0, BLOCK_M)
    block_start_idx = tl.load(block_start_idx_ptr + M_block_id)
    M_block = tl.max_contiguous(block_start_idx + M_range, BLOCK_M)
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_block < FAN_OUT * M,
        other=E)
    E_idx = tl.min(E_idxs)
    E_mask = E_idxs == E_idx
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=E_mask, other=0)
    if x_grouped:
        M_in_idx = M_block
    else:
        M_in_idx = M_idx // FAN_OUT
    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx
    K_block = tl.arange(0, BLOCK_K)
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :
        ] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :
        ] * stride_wn + E_idx * stride_we
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    iters = tl.cdiv(K, BLOCK_K)
    for K_block_id in range(0, iters):
        if NO_K_MASK:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            if NO_N_MASK or K_block_id < iters - 1:
                w = tl.load(W_blk_ptrs)
            else:
                w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = K_block_id * BLOCK_K + K_block < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        acc += tl.dot(x, w, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)
    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] *
        stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :])
