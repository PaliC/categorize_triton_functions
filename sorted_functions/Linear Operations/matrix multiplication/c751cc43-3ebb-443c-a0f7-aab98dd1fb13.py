import triton
import triton.language as tl
import torch

@triton.jit
def _reg_matmul(pid_n, type_id, start_off, input, other, output, N,
    stride_input_m, stride_input_k, stride_other_b, stride_other_k,
    stride_other_n, stride_output_m, stride_output_n, out_dtype:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', EVEN_N: 'tl.constexpr',
    TILE_M: 'tl.constexpr', TILE_N: 'tl.constexpr', TILE_K: 'tl.constexpr'):
    offs_m = start_off + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    rn = tl.max_contiguous(tl.multiple_of(offs_n % N, TILE_N), TILE_N)
    other_ptrs = other + type_id * stride_other_b + (offs_k[:, None] *
        stride_other_k + rn[None, :] * stride_other_n)
    b = tl.load(other_ptrs)
    input_ptrs = input + (offs_m[:, None] * stride_input_m + offs_k[None, :
        ] * stride_input_k)
    output_ptrs = output + stride_output_m * offs_m[:, None
        ] + stride_output_n * offs_n[None, :]
    for _ in range(0, BLOCK_SIZE):
        a = tl.load(input_ptrs)
        acc = tl.dot(a, b, out_dtype=out_dtype)
        if EVEN_N:
            tl.store(output_ptrs, acc)
        else:
            mask_n = offs_n[None, :] < N
            tl.store(output_ptrs, acc, mask=mask_n)
        input_ptrs += TILE_M * stride_input_m
        output_ptrs += TILE_M * stride_output_m
