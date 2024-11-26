import triton
import triton.language as tl
import torch

@triton.jit
def _reduce(c_ptr, c_buf_ptr, M, N, stride_cm, stride_cn, stride_cb_m,
    stride_cb_n, stride_cb_k, PK: 'tl.constexpr', BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_n
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, PK)
    c_buf_ptrs = c_buf_ptr + (offs_m[:, None, None] * stride_cb_m + offs_n[
        None, :, None] * stride_cb_n + offs_k[None, None, :] * stride_cb_k)
    c_buf = tl.load(c_buf_ptrs)
    reduced_k = tl.sum(c_buf, axis=2)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        )
    tl.store(c_ptrs, reduced_k)
