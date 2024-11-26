import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'TILE_SIZE_M': 64, 'TILE_SIZE_N': 
    32, 'TILE_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=5, num_warps=4)],
    key=['N', 'C', 'H', 'W', 'R', 'S', 'K'])
@triton.jit
def implicit_gemm_fprop_kernel(a_ptr, b_ptr, c_ptr, N, C, H, W, R, S, K,
    stride_An, stride_Ah, stride_Aw, stride_Ac, stride_Bk, stride_Br,
    stride_Bs, stride_Bc, stride_Cn, stride_Cp, stride_Cq, stride_Ck,
    TILE_SIZE_M: 'tl.constexpr', TILE_SIZE_N: 'tl.constexpr', TILE_SIZE_K:
    'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr'):
    Pad_H = 1
    Pad_W = 1
    Stride_H = 1
    Stride_W = 1
    Dilation_H = 1
    Dilation_W = 1
    P = (H + Pad_H * 2 - R * Dilation_H) // Stride_H + 1
    Q = (W + Pad_W * 2 - S * Dilation_W) // Stride_W + 1
    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, TILE_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, TILE_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    pq = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    q = pq % Q
    p = pq // Q
    k = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)
    crs = tl.arange(0, TILE_SIZE_K)
    s = crs % S
    c = crs // S // R
    r = crs // S % R
    a_ptrs = a_ptr + q[:, None] * stride_Aw + p[:, None] * stride_Ah + r[
        None, :] * stride_Ah + s[None, :] * stride_Aw + c[None, :] * stride_Ac
    b_ptrs = b_ptr + r[:, None] * stride_Br + s[:, None] * stride_Bs + c[:,
        None] * stride_Bc + k[None, :] * stride_Bk
    accumulator = tl.zeros((TILE_SIZE_M, TILE_SIZE_N), dtype=tl.float32)
    for gemm_k in range(0, GEMM_K, TILE_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        crs += TILE_SIZE_K
        s = crs % S
        c = crs // S // R
        r = crs // S % R
        a_ptrs = a_ptr + q[:, None] * stride_Aw + p[:, None] * stride_Ah + r[
            None, :] * stride_Ah + s[None, :] * stride_Aw + c[None, :
            ] * stride_Ac
        b_ptrs = b_ptr + r[:, None] * stride_Br + s[:, None] * stride_Bs + c[
            :, None] * stride_Bc + k[None, :] * stride_Bk
    c = accumulator
    offs_cm = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    offs_cn = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)
    c_ptrs = c_ptr + stride_Cq * offs_cm[:, None] + stride_Ck * offs_cn[None, :
        ]
    c_mask = (offs_cm[:, None] < GEMM_M) & (offs_cn[None, :] < GEMM_N)
    tl.store(c_ptrs, c, mask=c_mask)
