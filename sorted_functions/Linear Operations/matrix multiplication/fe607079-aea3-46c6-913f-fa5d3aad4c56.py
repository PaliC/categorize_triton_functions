import triton
import triton.language as tl
import torch

@triton.autotune(configs=int8_dynamic_configs(), key=['M', 'N', 'K'],
    prune_configs_by={'early_config_prune': early_config_prune,
    'perf_model': estimate_matmul_time, 'top_k': 10})
@triton.heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] *
    args['SPLIT_K']) == 0})
@triton.jit
def _kernel(A, B, C, bias, M, N, K, stride_am, stride_ak, stride_bk,
    stride_bn, stride_cm, stride_cn, a_scale_ptr, b_scale_ptr, out_dtype:
    'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    BLOCK_K: 'tl.constexpr', GROUP_M: 'tl.constexpr', SPLIT_K:
    'tl.constexpr', EVEN_K: 'tl.constexpr', BIAS_ADD: 'tl.constexpr',
    A_PER_CHANNEL: 'tl.constexpr', B_PER_CHANNEL: 'tl.constexpr'):
    """Kernel for computing the matmul C = A x B (optional + bias).
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    if A_PER_CHANNEL:
        a_scale = tl.load(a_scale_ptr + ram)
    else:
        a_scale = tl.load(a_scale_ptr)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            _0 = tl.zeros((1, 1), dtype=tl.int8)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if A_PER_CHANNEL:
            a = tl.math.llrint(a / a_scale[:, None])
        else:
            a = tl.math.llrint(a / a_scale)
        acc += tl.dot(a, b, allow_tf32=True, out_dtype=tl.int32)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    if B_PER_CHANNEL:
        b_scale = tl.load(b_scale_ptr + rbn)
    else:
        b_scale = tl.load(b_scale_ptr)
    if A_PER_CHANNEL and B_PER_CHANNEL:
        acc = acc.to(tl.float32) * (a_scale[:, None] * b_scale[None, :])
    else:
        acc = acc.to(tl.float32) * (a_scale * b_scale)
    if BIAS_ADD:
        bias = tl.load(bias + rn)
        acc = acc + bias[None, :]
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)
