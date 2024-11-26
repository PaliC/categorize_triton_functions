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
@triton.heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] *
    args['SPLIT_K']) == 0})
@triton.jit
def kernel_bwd(C, ACT_INPUT, A, B, M, N, K, CACHE_KEY_M, CACHE_KEY_N,
    CACHE_KEY_K, stride_cm, stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_M: 'tl.constexpr', GROUP_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', BLOCK_K: 'tl.constexpr', SPLIT_K: 'tl.constexpr',
    EVEN_K: 'tl.constexpr', ACTIVATION: 'tl.constexpr'):
    """
    Kernel for computing Out = activation(A x W + C)
    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)
    'ActInputs' optionally saves the A x W + C intermediate for backward computations
    This kernel will consolidate over K
    """
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    if ACTIVATION != 'id':
        act_in_ptrs = ACT_INPUT + ram[:, None] * stride_cm + rbn[None, :]
        act_input = tl.load(act_in_ptrs)
    if ACTIVATION == 'gelu':
        acc *= gelu_grad(act_input)
    elif ACTIVATION == 'gelu_approx':
        acc *= gelu_approx_grad(act_input)
    elif ACTIVATION == 'squared_relu':
        acc *= squared_relu_grad(act_input)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + rm[:, None] * stride_cm + rn[None, :]
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)
