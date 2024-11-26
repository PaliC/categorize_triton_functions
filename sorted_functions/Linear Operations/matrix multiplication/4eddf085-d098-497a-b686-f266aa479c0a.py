import triton
import triton.language as tl
import torch

@triton.autotune(configs=configs, key=['CACHE_KEY_M', 'CACHE_KEY_N',
    'BATCHSIZE', 'SPARSITY_BIN'])
@triton.jit
def splitk_sparse_gemv_kernel(Y, A, X, threshold, N, M, CACHE_KEY_N,
    CACHE_KEY_M, BATCHSIZE: 'tl.constexpr', SPARSITY_BIN: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr'):
    start_n = tl.program_id(0)
    start_m = tl.program_id(1)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    A_ptr = A + (rm[:, None] * N + rn[None, :])
    X_ptr = X + rm
    Y_ptr = Y + rn
    if BATCHSIZE == 1:
        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy=
            'evict_last')
        idx = tl.abs(x0) > threshold
        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy=
            'evict_first')
        acc0 = tl.sum(a * x0[:, None], 0)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y_ptr, acc0, mask=rn < N)
