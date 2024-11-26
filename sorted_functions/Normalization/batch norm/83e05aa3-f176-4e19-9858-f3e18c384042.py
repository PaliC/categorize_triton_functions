import triton
import triton.language as tl
import torch

@eval(
    """triton.heuristics({
    'BLOCK_SIZE':
    lambda kwargs: triton.next_power_of_2(kwargs['cluster_num']),
})"""
    )
@eval(
    """triton.heuristics({
    'num_warps':
    lambda kwargs: max(1, min(16, kwargs['BLOCK_SIZE'] // 128)),
})"""
    )
@triton.jit
def group_norm_4d_channels_last_forward_collect_stats_kernel_stage_2(
    cluster_mean_ptr, cluster_m2_ptr, cluster_weight_ptr, N, groups,
    cluster_num, eps, mean_ptr, rstd_ptr, BLOCK_SIZE: 'tl.constexpr'):
    group = tl.program_id(0)
    pid_batch = tl.program_id(1)
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < cluster_num
    offset = pid_batch * groups * cluster_num + group * cluster_num + block
    cluster_mean = tl.load(cluster_mean_ptr + offset, mask=mask)
    cluster_m2 = tl.load(cluster_m2_ptr + offset, mask=mask)
    cluster_weight = tl.load(cluster_weight_ptr + offset, mask=mask)
    mean, m2, weight = tl.reduce((cluster_mean, cluster_m2, cluster_weight),
        0, welford_combine)
    var = m2 / weight
    rstd = 1.0 / tl.sqrt(var + eps)
    offset = pid_batch * groups + group
    tl.store(mean_ptr + offset, mean)
    tl.store(rstd_ptr + offset, rstd)
