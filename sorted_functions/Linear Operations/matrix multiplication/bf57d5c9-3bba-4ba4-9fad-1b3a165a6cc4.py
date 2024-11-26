import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256,
    'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config
    ({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 
    64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 
    128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,
    'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2), triton.Config
    ({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 
    128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.
    Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 
    256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 
    32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.
    Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1},
    num_stages=5, num_warps=2)] + get_configs_io_bound(), key=[
    'CACHE_KEY_M', 'CACHE_KEY_N', 'CACHE_KEY_K'], prune_configs_by={
    'early_config_prune': early_config_prune, 'perf_model':
    estimate_matmul_time, 'top_k': 10})
@triton.heuristics({'K_LOAD_MASK_NEEDED': lambda args: args['K'] % (args[
    'BLOCK_K'] * args['SPLIT_K']) == 0})
@triton.jit
def kernel_fma(C, A, B, bias, dtype: 'tl.constexpr', M, N, K, CACHE_KEY_M,
    CACHE_KEY_N, CACHE_KEY_K, output_m_stride, output_n_stride, a_m_stride,
    a_k_stride, b_n_stride, b_k_stride, BLOCK_M: 'tl.constexpr', GROUP_M:
    'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr',
    SPLIT_K: 'tl.constexpr', K_LOAD_MASK_NEEDED: 'tl.constexpr', HAS_BIAS:
    'tl.constexpr', ACTIVATION: 'tl.constexpr'):
    """
    Kernel for computing Out = activation(A x W + C)

    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)

    'ActInputs' optionally saves the A x W + C intermediate for backward computations

    This kernel will consolidate over K
    """
    program_idx = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = GROUP_M * grid_n
    group_idx = program_idx // width
    group_size = min(grid_m - group_idx * GROUP_M, GROUP_M)
    block_m_idx = group_idx * GROUP_M + program_idx % group_size
    block_n_idx = program_idx % width // group_size
    m_offs_untagged = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs_untagged = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    m_offs = tl.max_contiguous(tl.multiple_of(m_offs_untagged % M, BLOCK_M),
        BLOCK_M)
    n_offs = tl.max_contiguous(tl.multiple_of(n_offs_untagged % N, BLOCK_N),
        BLOCK_N)
    k_range_offs = tl.arange(0, BLOCK_K)
    A = A + (m_offs[:, None] * a_m_stride + k_range_offs[None, :] * a_k_stride)
    B = B + (k_range_offs[:, None] * b_k_stride + n_offs[None, :] * b_n_stride)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias + n_offs, mask=n_offs < N, other=0.0)
        acc += bias[None, :]
    for k in range(K, 0, -BLOCK_K):
        if K_LOAD_MASK_NEEDED:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=k_range_offs[None, :] < k, other=0.0)
            b = tl.load(B, mask=k_range_offs[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * a_k_stride
        B += BLOCK_K * b_k_stride
    if ACTIVATION:
        acc = silu(acc)
    acc = acc
    C = C + m_offs[:, None] * output_m_stride + n_offs[None, :
        ] * output_n_stride
    c_ptr_mask = (m_offs < M)[:, None] & (n_offs < N)[None, :]
    tl.store(C, acc, mask=c_ptr_mask)
