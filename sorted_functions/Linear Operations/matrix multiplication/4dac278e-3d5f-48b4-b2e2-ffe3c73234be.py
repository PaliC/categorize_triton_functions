import triton
import triton.language as tl
import torch

@triton.jit
def _single2scatter(X_ptr, stride_xm, stride_xk, W_ptr, stride_we,
    stride_wk, stride_wn, Y_ptr, stride_ym, stride_yn, expert_idxs_ptr,
    FAN_OUT: 'tl.constexpr', K: 'tl.constexpr', N: 'tl.constexpr', E:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr',
    ACC_TYPE: 'tl.constexpr'):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    N_block_id = pid0
    if FAN_OUT == 1:
        in_idx = pid1
    else:
        in_idx = 0
    out_idx = pid1
    K_block = tl.arange(0, BLOCK_K)
    N_block = tl.max_contiguous(tl.multiple_of((N_block_id * BLOCK_N + tl.
        arange(0, BLOCK_N)) % N, BLOCK_N), BLOCK_N)
    E_idx = tl.load(expert_idxs_ptr + pid1)
    X_blk_ptrs = X_ptr + in_idx * stride_xm + K_block[:, None] * stride_xk
    W_blk_ptrs = W_ptr + E_idx * stride_we + K_block[:, None
        ] * stride_wk + N_block[None, :] * stride_wn
    acc = tl.zeros((1, BLOCK_N), dtype=ACC_TYPE)
    for K_block_id in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(X_blk_ptrs)
        w = tl.load(W_blk_ptrs)
        acc += tl.sum(x * w, axis=0)[None, :]
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
    Y_blk_ptrs = Y_ptr + out_idx * stride_ym + N_block[None, :] * stride_yn
    tl.store(Y_blk_ptrs, acc)
