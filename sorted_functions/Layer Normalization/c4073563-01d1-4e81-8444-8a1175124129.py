import triton
import triton.language as tl
import torch

@eval(
    """triton.heuristics({
    'ROW_SIZE':
    lambda kwargs: triton.next_power_of_2(kwargs['C'] // kwargs['groups']),
    'BLOCK_SIZE':
    lambda kwargs: max(
        1, min(triton.next_power_of_2(kwargs['HxW']),
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
def group_norm_4d_channels_last_forward_collect_stats_kernel(input_ptr, N,
    C, HxW, groups, eps, mean_ptr, rstd_ptr, C_G, ROW_SIZE: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr'):
    group = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offset = pid_batch * C * HxW + group * C_G
    X = input_ptr + offset
    _mean = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _m2 = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _weight = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    row = tl.arange(0, ROW_SIZE)
    for off in range(0, HxW, BLOCK_SIZE):
        r = off + tl.arange(0, BLOCK_SIZE)
        m2_ = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
        mask = (r < HxW)[:, None] & (row[None, :] < C_G)
        weight_ = mask
        x = tl.load(X + (r * C)[:, None] + row[None, :], mask=mask)
        _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x, m2_,
            weight_)
    _mean = tl.view(_mean, (BLOCK_SIZE * ROW_SIZE,))
    _m2 = tl.view(_m2, (BLOCK_SIZE * ROW_SIZE,))
    _weight = tl.view(_weight, (BLOCK_SIZE * ROW_SIZE,))
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1.0 / tl.sqrt(var + eps)
    offset = pid_batch * groups + group
    tl.store(mean_ptr + offset, mean)
    tl.store(rstd_ptr + offset, rstd)
