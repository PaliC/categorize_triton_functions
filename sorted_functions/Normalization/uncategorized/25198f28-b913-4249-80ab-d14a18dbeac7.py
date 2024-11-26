import triton
import triton.language as tl
import torch

@eval(
    """triton.heuristics({
    'ROW_SIZE':
    lambda kwargs: triton.next_power_of_2(kwargs['C'] // kwargs['groups']),
    'BLOCK_SIZE':
    lambda kwargs: max(
        1, min(triton.next_power_of_2(kwargs['cluster_size']),
               4096 // (triton.next_power_of_2(kwargs['C'] // kwargs['groups']))
               )),
})"""
    )
@eval(
    """triton.heuristics({
    'num_warps':
    lambda kwargs: max(1, min(16, kwargs['ROW_SIZE'] * kwargs['BLOCK_SIZE'] // 128)),
    'C_G': lambda kwargs: kwargs['C'] // kwargs['groups'],
})"""
    )
@triton.jit
def group_norm_4d_channels_last_forward_collect_stats_kernel_stage_1(input_ptr,
    N, C, HxW, groups, cluster_size, cluster_num, cluster_mean_ptr,
    cluster_m2_ptr, cluster_weight_ptr, C_G, ROW_SIZE: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr'):
    group = tl.program_id(0)
    cluster = tl.program_id(1)
    pid_batch = tl.program_id(2)
    offset = pid_batch * C * HxW + group * C_G
    X = input_ptr + offset
    _mean = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _m2 = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _weight = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    row = tl.arange(0, ROW_SIZE)
    start = cluster * cluster_size
    end = start + cluster_size
    end = min(end, HxW)
    for off in range(start, end, BLOCK_SIZE):
        r = off + tl.arange(0, BLOCK_SIZE)
        m2_ = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
        mask = (r < end)[:, None] & (row[None, :] < C_G)
        weight_ = mask
        x = tl.load(X + (r * C)[:, None] + row[None, :], mask=mask)
        _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x, m2_,
            weight_)
    _mean = tl.view(_mean, (BLOCK_SIZE * ROW_SIZE,))
    _m2 = tl.view(_m2, (BLOCK_SIZE * ROW_SIZE,))
    _weight = tl.view(_weight, (BLOCK_SIZE * ROW_SIZE,))
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    offset = pid_batch * groups * cluster_num + group * cluster_num + cluster
    tl.store(cluster_mean_ptr + offset, mean)
    tl.store(cluster_m2_ptr + offset, m2)
    tl.store(cluster_weight_ptr + offset, weight)
