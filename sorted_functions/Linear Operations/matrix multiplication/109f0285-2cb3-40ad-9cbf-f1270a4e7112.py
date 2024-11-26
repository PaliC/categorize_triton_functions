import triton
import triton.language as tl
import torch

@triton.jit
def streamk_kernel(a_ptr, b_ptr, c_ptr, M: 'tl.constexpr', N:
    'tl.constexpr', K: 'tl.constexpr', scratchpad, locks, stride_am:
    'tl.constexpr', stride_ak: 'tl.constexpr', stride_bk: 'tl.constexpr',
    stride_bn: 'tl.constexpr', stride_cm: 'tl.constexpr', stride_cn:
    'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N:
    'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr', USE_ATOMICS: 'tl.constexpr'):
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_iters = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N
        ) * iters_per_tile
    iters_per_cta = tl.cdiv(total_iters, tl.num_programs(0))
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    itr = pid * iters_per_cta
    iter_end = itr + iters_per_cta
    GROUP_SIZE_M = 8
    while itr < iter_end and itr < total_iters:
        tile_idx = itr // iters_per_tile
        tile_iter = tile_idx * iters_per_tile
        tile_iter_end = tile_iter + iters_per_tile
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_idx // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = tile_idx % num_pid_m
        pid_n = tile_idx // num_pid_m
        pid_k = itr - tile_iter
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
            stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] *
            stride_bn)
        local_iter = pid_k
        local_iter_end = min(iter_end, tile_iter_end) - tile_iter
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, local_iter_end - local_iter):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        acc = acc
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tile_started = itr == tile_iter
        tile_ended = iter_end >= tile_iter_end
        scratch_off = pid * BLOCK_SIZE_M * BLOCK_SIZE_N
        offs_scratch = tl.arange(0, BLOCK_SIZE_M)[:, None
            ] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
        if USE_ATOMICS:
            tl.atomic_add(c_ptr + c_offs, acc, c_mask)
        elif not tile_started:
            tl.store(scratchpad + scratch_off + offs_scratch, acc, c_mask)
            tl.atomic_xchg(locks + pid, 1)
        else:
            if not tile_ended:
                cta_end = tl.cdiv(tile_iter_end, iters_per_cta)
                cas = pid + 1
                while cas < cta_end:
                    while tl.atomic_cas(locks + cas, 1, 2) != 1:
                        pass
                    acc += tl.load(scratchpad + cas * BLOCK_SIZE_M *
                        BLOCK_SIZE_N + offs_scratch, c_mask)
                    cas += 1
            tl.store(c_ptr + c_offs, acc, c_mask)
        itr = tile_iter_end
