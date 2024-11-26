import triton
import triton.language as tl
import torch

@triton.jit
def _prefetch_matmul(pid_n, start_off, end_off, input, other, output, K, N,
    stride_input_m, stride_input_k, stride_other_k, stride_other_n,
    stride_output_m, stride_output_n, out_dtype: 'tl.constexpr', TILE_M:
    'tl.constexpr', TILE_N: 'tl.constexpr', TILE_K: 'tl.constexpr', EVEN_N:
    'tl.constexpr', EVEN_K: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    offs_m = start_off + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    rn = tl.max_contiguous(tl.multiple_of(offs_n % N, TILE_N), TILE_N)
    input_ptrs = input + (offs_m[:, None] * stride_input_m + offs_k[None, :
        ] * stride_input_k)
    other_ptrs = other + (offs_k[:, None] * stride_other_k + rn[None, :] *
        stride_other_n)
    output_ptrs = output + stride_output_m * offs_m[:, None
        ] + stride_output_n * offs_n[None, :]
    original_input_ptrs = input_ptrs
    original_other_ptrs = other_ptrs
    acc = tl.zeros((TILE_M, TILE_N), dtype=out_dtype)
    mask_n = offs_n[None, :] < N
    k_iters = K // TILE_K if EVEN_K else tl.cdiv(K, TILE_K)
    for k in range(0, k_iters * BLOCK_SIZE):
        i = k % k_iters
        if EVEN_K:
            a = tl.load(input_ptrs)
            b = tl.load(other_ptrs)
        else:
            a = tl.load(input_ptrs, mask=offs_k[None, :] + i * TILE_K < K,
                other=0.0)
            b = tl.load(other_ptrs, mask=offs_k[:, None] + i * TILE_K < K,
                other=0.0)
        acc += tl.dot(a, b, out_dtype=out_dtype)
        if i == k_iters - 1:
            if EVEN_N:
                tl.store(output_ptrs, acc)
            else:
                tl.store(output_ptrs, acc, mask_n)
            output_ptrs += TILE_M * stride_output_m
        if i == k_iters - 1:
            acc = tl.zeros((TILE_M, TILE_N), dtype=out_dtype)
            original_input_ptrs += TILE_M * stride_input_m
            input_ptrs = original_input_ptrs
            other_ptrs = original_other_ptrs
        else:
            input_ptrs += TILE_K * stride_input_k
            other_ptrs += TILE_K * stride_other_k
