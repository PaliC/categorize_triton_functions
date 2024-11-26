import triton
import triton.language as tl
import torch

@triton.jit
def _triton_gemm_a16w8_sub_channel_kernel(A, B, C, scale_b, bias,
    zero_points, M, N, K, stride_am, stride_ak, stride_bn, stride_bk,
    stride_cm, stride_cn, stride_zpk, stride_zpn, stride_scalek,
    stride_scalen, add_bias: 'tl.constexpr', add_zero_points:
    'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    BLOCK_K: 'tl.constexpr', GROUP_M: 'tl.constexpr', SPLIT_K: 'tl.constexpr'):
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale_w_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    _SCALE0 = tl.zeros([1], dtype=scale_b.dtype.element_ty)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)
        _A0 = tl.zeros((1, 1), dtype=A.dtype.element_ty)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_A0)
        _B0 = tl.zeros((1, 1), dtype=B.dtype.element_ty)
        b = tl.load(B, mask=rk[None, :] < k_remaining, other=_B0)
        if add_zero_points:
            _ZERO_POINT0 = tl.zeros([1], dtype=zero_points.dtype.element_ty)
            zero_points_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            zero_points_ptrs = zero_points + (k * SPLIT_K + pid_z
                ) * stride_zpk + zero_points_offs
            zero_points_vals = tl.load(zero_points_ptrs, mask=
                zero_points_offs < N, other=_ZERO_POINT0)
            b = b - zero_points_vals[:, None]
        scale_ptrs = (scale_b + k * SPLIT_K * stride_scalek + pid_z *
            stride_scalek + scale_w_offs)
        scales = tl.load(scale_ptrs, mask=scale_w_offs < N, other=_SCALE0)
        b_fp = b * scales[:, None]
        b_fp = tl.trans(b_fp)
        acc += tl.dot(a, b_fp, out_dtype=tl.float32, allow_tf32=True)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if add_bias:
        offs_bias = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_ptrs = bias + offs_bias
        _BIAS0 = tl.zeros([1], dtype=bias.dtype.element_ty)
        bias_vals = tl.load(bias_ptrs, mask=offs_bias < N, other=_BIAS0)
        if pid_z == 0:
            acc += bias_vals[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)
