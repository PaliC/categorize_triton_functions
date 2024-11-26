import triton
import triton.language as tl
import torch

@triton.jit
def _triton_gemm_a16w4_per_channel_kernel(A, B, C, scale_b, bias,
    zero_points, M, N, K, rescale_m, rescale_n, rescale_k, stride_am,
    stride_ak, stride_bn, stride_bk, stride_cm, stride_cn, stride_zpk,
    stride_zpn, stride_scalek, stride_scalen, add_bias: 'tl.constexpr',
    add_zero_points: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', BLOCK_K: 'tl.constexpr', GROUP_M: 'tl.constexpr',
    SPLIT_K: 'tl.constexpr'):
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
    B = B + (rbn[:, None] * stride_bn + rk[None, :] * stride_bk)
    acc_l = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    acc_h = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    _A0 = tl.zeros((1, 1), dtype=A.dtype.element_ty)
    _B0 = tl.zeros((1, 1), dtype=B.dtype.element_ty)
    if add_zero_points:
        offs_zero_points = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
        zero_points_ptrs = zero_points + offs_zero_points
        _ZERO_POINT0 = tl.zeros([1], dtype=zero_points.dtype.element_ty)
        zero_points_vals = tl.load(zero_points_ptrs, mask=offs_zero_points <
            2 * N, other=_ZERO_POINT0)
        zero_points_vals = tl.reshape(zero_points_vals, (BLOCK_N, 2))
        zp_l, zp_h = tl.split(zero_points_vals)
    offs_scale = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    scale_ptrs = scale_b + offs_scale
    _SCALE0 = tl.zeros([1], dtype=scale_b.dtype.element_ty)
    scales = tl.load(scale_ptrs, mask=offs_scale < 2 * N, other=_SCALE0)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)
        b_int4_two = tl.load(B, mask=rk[None, :] < k_remaining, other=_B0)
        b_int4_l = b_int4_two.__lshift__(4).to(tl.int8).__rshift__(4)
        b_int4_h = b_int4_two.__rshift__(4)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_A0)
        a = tl.trans(a)
        if add_zero_points:
            b_int4_l -= zp_l[:, None]
            b_int4_h -= zp_h[:, None]
        acc_l += tl.dot(b_int4_l, a, out_dtype=tl.float32, allow_tf32=True)
        acc_h += tl.dot(b_int4_h, a, out_dtype=tl.float32, allow_tf32=True)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc_l = tl.trans(acc_l)
    acc_h = tl.trans(acc_h)
    acc = tl.interleave(acc_l, acc_h)
    offs_scale = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    scale_ptrs = scale_b + offs_scale
    _SCALE0 = tl.zeros([1], dtype=scale_b.dtype.element_ty)
    scales = tl.load(scale_ptrs, mask=offs_scale < 2 * N, other=_SCALE0)
    acc *= scales[None, :]
    acc = acc
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
    mask = (rm < M)[:, None] & (rn < 2 * N)[None, :]
    if add_bias:
        offs_bias = pid_n * BLOCK_N * 2 + tl.arange(0, 2 * BLOCK_N)
        bias_ptrs = bias + offs_bias
        _BIAS0 = tl.zeros([1], dtype=bias.dtype.element_ty)
        bias_vals = tl.load(bias_ptrs, mask=offs_bias < 2 * N, other=_BIAS0)
        if pid_z == 0:
            acc += bias_vals[None, :]
    if SPLIT_K == 1:
        tl.store(C + rm[:, None] * stride_cm + rn[None, :], acc, mask=mask)
    else:
        tl.atomic_add(C + rm[:, None] * stride_cm + rn[None, :], acc, mask=mask
            )
