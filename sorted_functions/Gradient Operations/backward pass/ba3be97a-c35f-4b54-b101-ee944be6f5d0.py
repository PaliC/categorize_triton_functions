import triton
import triton.language as tl
import torch

@triton.jit
def bias_kernel_backward(d_weights, d_out, weights, stride_om, stride_on,
    stride_wn, N: 'tl.constexpr', M: 'tl.constexpr', NH: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_NH:
    'tl.constexpr', BIDIRECTIONAL: 'tl.constexpr', NUM_BUCKETS:
    'tl.constexpr', MAX_DISTANCE: 'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr'
    ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    relative_positions = offs_m[:, None] - offs_n[None, :]
    relative_buckets = tl.zeros_like(relative_positions)
    num_buckets = NUM_BUCKETS
    if BIDIRECTIONAL:
        num_buckets //= 2
        relative_buckets += (relative_positions > 0) * num_buckets
        relative_positions = tl.abs(relative_positions)
    else:
        relative_positions = tl.maximum(-relative_positions, tl.zeros_like(
            relative_positions))
    max_exact = num_buckets // 2
    is_small = relative_positions < max_exact
    relative_position_if_large = max_exact + tl.log(relative_positions.to(
        tl.float32) / max_exact) / tl.log(MAX_DISTANCE / max_exact) * (
        num_buckets - max_exact)
    relative_position_if_large = tl.minimum(relative_position_if_large, 
        num_buckets - 1)
    relative_buckets += tl.where(is_small, relative_positions,
        relative_position_if_large)
    for i in range(0, NH, BLOCK_NH):
        offs_nh = i + tl.arange(0, BLOCK_NH)
        bucket_offs = relative_buckets[:, :, None] * stride_wn + offs_nh[
            None, None, :]
        d_out_ptrs = d_out + (offs_m[:, None] * stride_om + offs_n[None, :] *
            stride_on)[:, :, None] + offs_nh[None, None, :]
        o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)[:, :, None] & (
            offs_nh[None, None, :] < NH)
        d_out_values = tl.load(d_out_ptrs, mask=o_mask, other=0.0)
        d_weights_ptrs = d_weights + bucket_offs
        tl.atomic_add(d_weights_ptrs, d_out_values, mask=relative_buckets[:,
            :, None] < NUM_BUCKETS)
