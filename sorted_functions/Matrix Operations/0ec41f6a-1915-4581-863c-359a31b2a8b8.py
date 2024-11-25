import triton
import triton.language as tl
import torch

@triton.jit
def fused_moe_kernel(a_ptr, b_ptr, c_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, N, K,
    EM, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk,
    stride_bn, stride_cm, stride_cn, stride_weight, stride_token_id,
    block_m: 'tl.constexpr', block_n: 'tl.constexpr', block_k:
    'tl.constexpr', MUL_ROUTED_WEIGHT: 'tl.constexpr', top_k:
    'tl.constexpr', compute_type: 'tl.constexpr'):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can be any shape representing batches and K is the feature dimension of each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is the number of experts, K is the input feature dimension, and N is the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the total number of tokens post padding, topk is the number of times each token is repeated,
        and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens, repeated topk times and arranged by the expert index they are assigned to.
    - expert_ids: A tensor containing the indices of the expert for each block. It determines which expert matrix from B should be used for each block in A.
    This kernel performs the multiplication of a token by its corresponding expert matrix as determined by `expert_ids`. The sorting of `sorted_token_ids`
    by expert index and padding ensures divisibility by block_m, which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert.
    """
    pid = tl.program_id(axis=0)
    pid_m, pid_n = col_major(pid, EM, N, block_m, block_n)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * block_m >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * block_m + tl.arange(0, block_m)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    offs_bn = (pid_n * block_n + tl.arange(0, block_n)) % N
    offs_k = tl.arange(0, block_k)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[
        None, :] * stride_ak)
    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
        offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, block_k)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K -
            k * block_k), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * block_k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_weight,
            mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
