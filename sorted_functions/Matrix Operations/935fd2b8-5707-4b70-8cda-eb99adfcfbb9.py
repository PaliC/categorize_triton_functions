import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel_tma(a_desc_ptr, b_desc_ptr, c_desc_ptr, prob_m, prob_n,
    prob_k, block_m: 'tl.constexpr', block_n: 'tl.constexpr', block_k:
    'tl.constexpr'):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(prob_m, block_m)
    num_pid_k = tl.cdiv(prob_k, block_k)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * block_m
    offs_bn = pid_n * block_n
    offs_k = 0
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for kk in range(0, num_pid_k):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k],
            [block_m, block_k], tl.float8e4nv)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k],
            [block_n, block_k], tl.float8e4nv)
        accumulator = tl.dot(a, b.T, acc=accumulator, out_dtype=tl.float32)
        offs_k += block_k
    accumulator = accumulator
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am,
        offs_bn])
