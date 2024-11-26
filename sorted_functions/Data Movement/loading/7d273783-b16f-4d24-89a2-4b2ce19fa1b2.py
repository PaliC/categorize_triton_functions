import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128},
    num_warps=2, pre_hook=init_to_zero('Y')), triton.Config({'BLOCK_M': 16,
    'BLOCK_N': 256}, num_warps=4, pre_hook=init_to_zero('Y')), triton.
    Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=4, pre_hook=
    init_to_zero('Y')), triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512},
    num_warps=4, pre_hook=init_to_zero('Y')), triton.Config({'BLOCK_M': 16,
    'BLOCK_N': 1024}, num_warps=4, pre_hook=init_to_zero('Y')), triton.
    Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4, pre_hook=
    init_to_zero('Y')), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512},
    num_warps=4, pre_hook=init_to_zero('Y')), triton.Config({'BLOCK_M': 32,
    'BLOCK_N': 1024}, num_warps=4, pre_hook=init_to_zero('Y')), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=4, pre_hook=
    init_to_zero('Y')), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512},
    num_warps=4, pre_hook=init_to_zero('Y')), triton.Config({'BLOCK_M': 64,
    'BLOCK_N': 1024}, num_warps=4, pre_hook=init_to_zero('Y')), triton.
    Config({'BLOCK_M': 128, 'BLOCK_N': 16}, num_warps=4, pre_hook=
    init_to_zero('Y')), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},
    num_warps=4, pre_hook=init_to_zero('Y')), triton.Config({'BLOCK_M': 128,
    'BLOCK_N': 64}, num_warps=4, pre_hook=init_to_zero('Y')), triton.Config
    ({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, pre_hook=init_to_zero(
    'Y')), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=4,
    pre_hook=init_to_zero('Y')), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    512}, num_warps=4, pre_hook=init_to_zero('Y')), triton.Config({
    'BLOCK_M': 128, 'BLOCK_N': 1024}, num_warps=4, pre_hook=init_to_zero(
    'Y'))], key=['CACHE_KEY_M', 'CACHE_KEY_N', 'BATCHSIZE', 'SPARSITY_BIN'])
@triton.jit
def gather_transposed_gemv_flag_atomicadd_kernel(Y, A, X, IDX, M, N,
    CACHE_KEY_M, CACHE_KEY_N, stride_am, BATCHSIZE: 'tl.constexpr',
    SPARSITY_BIN: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr'):
    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0) > 0
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rm
    Y = Y + rn
    if BATCHSIZE == 1:
        a = tl.load(A, mask=idx[:, None], other=0.0)
        x0 = tl.load(X)
        acc0 = tl.sum(a * x0[:, None], 0)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y, acc0, mask=rn < N)
