import triton
import triton.language as tl
import torch

@triton.jit
def _fused_moe_a8w8_kernel(A, B, C, alpha_row_ptr, alpha_col_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr, N, K, EM, num_valid_tokens, stride_am,
    stride_ak, stride_be, stride_bn, stride_bk, stride_cm, stride_cn,
    stride_scale_be, stride_scale_bn, BLOCK_SIZE_M: 'tl.constexpr',
    BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr',
    GROUP_SIZE_M: 'tl.constexpr', MUL_ROUTED_WEIGHT: 'tl.constexpr', top_k:
    'tl.constexpr'):
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
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_token[:, None] // top_k * stride_am + offs_k[None, :
        ] * stride_ak)
    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = B + off_experts * stride_be + (offs_bn[None, :] * stride_bn + 
        offs_k[:, None] * stride_bk)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    _A0 = tl.zeros([1, 1], dtype=a_ptrs.dtype.element_ty)
    _B0 = tl.zeros([1, 1], dtype=b_ptrs.dtype.element_ty)
    lo = 0
    hi = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(lo, hi - 1):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=_A0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    for k in range(hi - 1, hi):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K -
            k * BLOCK_SIZE_K), other=_A0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=_B0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    alpha_row_ptrs = alpha_row_ptr + offs_token // top_k
    alpha_col_ptrs = alpha_col_ptr + off_experts * stride_scale_be + offs_cn
    _ALPHA0 = tl.zeros([1], dtype=alpha_row_ptr.dtype.element_ty)
    alpha_row = tl.load(alpha_row_ptrs, mask=token_mask, other=_ALPHA0)
    alpha_col = tl.load(alpha_col_ptrs, mask=offs_cn < N, other=_ALPHA0)
    accumulator = accumulator * alpha_row[:, None]
    accumulator = accumulator * alpha_col[None, :]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask,
            other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator
    c_ptrs = C + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
