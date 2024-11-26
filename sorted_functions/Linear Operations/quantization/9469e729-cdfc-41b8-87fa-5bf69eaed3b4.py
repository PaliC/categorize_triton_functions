import triton
import triton.language as tl
import torch

@triton.autotune(configs=configs, key=['CACHE_KEY_M', 'CACHE_KEY_N',
    'BATCHSIZE', 'SPARSITY_BIN'])
@triton.jit
def qkv_kernel(Y, A, X, threshold_q, threshold_k, threshold_v, N, N_q, N_kv,
    M, CACHE_KEY_N, CACHE_KEY_M, BATCHSIZE: 'tl.constexpr', SPARSITY_BIN:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr'):
    start_n = tl.program_id(0)
    start_m = tl.program_id(1)
    is_q = start_n * BLOCK_N < N_q
    is_v = N_q + N_kv <= start_n * BLOCK_N
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A_ptr = A + rm[:, None] * N + rn[None, :]
    X_ptr = X + rm
    Y_ptr = Y + rn
    threshold = tl.where(is_q, threshold_q, tl.where(is_v, threshold_v,
        threshold_k))
    if BATCHSIZE == 1:
        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy=
            'evict_last')
        idx = tl.abs(x0) > threshold
        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy=
            'evict_first')
        acc = tl.sum(a * x0[:, None], 0)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < N
    tl.atomic_add(Y_ptr, acc, mask=mask_n)
