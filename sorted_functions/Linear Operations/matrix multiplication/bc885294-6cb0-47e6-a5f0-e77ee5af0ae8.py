import triton
import triton.language as tl
import torch

@triton.jit
def _fused_moe_kernel_a16w4_perchannel(A, B, C, scale_b_ptr,
    zero_points_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr, N, K, EM, num_valid_tokens, stride_am,
    stride_ak, stride_be, stride_bn, stride_bk, stride_cm, stride_cn,
    stride_scale_be, stride_scale_bn, stride_scale_bk, stride_zero_points_e,
    stride_zero_points_n, stride_zero_points_k, BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K:
    'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr', MUL_ROUTED_WEIGHT:
    'tl.constexpr', top_k: 'tl.constexpr', add_zero_points: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N * 2) // 2) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_token[:, None] // top_k * stride_am + offs_k[None, :
        ] * stride_ak)
    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = B + off_experts * stride_be + (offs_k[None, :] * stride_bk + 
        offs_bn[:, None] * stride_bn)
    if add_zero_points:
        offs_zero_points = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, 2 *
            BLOCK_SIZE_N)
        zero_points_ptrs = (zero_points_ptr + off_experts *
            stride_zero_points_e + offs_zero_points)
        _ZERO_POINT0 = tl.zeros([1], dtype=zero_points_ptr.dtype.element_ty)
        zero_points_vals = tl.load(zero_points_ptrs, mask=offs_zero_points <
            2 * N, other=_ZERO_POINT0)
    _A0 = tl.zeros([1, 1], dtype=A.dtype.element_ty)
    _B0 = tl.zeros([1, 1], dtype=B.dtype.element_ty)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
    l_shifter = (1 - tl.arange(0, BLOCK_SIZE_N * 2) % 2) * 4
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K -
            k * BLOCK_SIZE_K), other=_A0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
            other=_B0)
        b = (b << l_shifter[:, None]).__rshift__(4)
        if add_zero_points:
            b -= zero_points_vals[:, None]
        b = tl.trans(b)
        b = b
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_scale = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, BLOCK_SIZE_N * 2)
    scale_ptrs = (scale_b_ptr + off_experts * stride_scale_be + offs_scale *
        stride_scale_bn)
    _SCALE0 = tl.zeros([1], dtype=scale_b_ptr.dtype.element_ty)
    scales = tl.load(scale_ptrs, mask=offs_scale < 2 * N, other=_SCALE0)
    accumulator *= scales[None, :]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask,
            other=0.0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator
    offs_cn = pid_n * BLOCK_SIZE_N * 2 + tl.arange(0, BLOCK_SIZE_N * 2)
    c_ptrs = C + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N * 2)
    tl.store(c_ptrs, accumulator, mask=c_mask)
