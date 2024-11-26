import triton
import triton.language as tl
import torch

@triton.jit
def fused_ffn_kernel(x_ptr, w_ptr, z_ptr, M, N, K, b_ptr=None, r_ptr=None,
    apply_gelu=False, dropout_prob=0.0, seed=1337, BLOCK_SIZE_M:
    'tl.constexpr'=128, BLOCK_SIZE_N: 'tl.constexpr'=128, BLOCK_SIZE_K:
    'tl.constexpr'=64):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k
        x = tl.load(x_ptr + offs_m * K + x_k, mask=(offs_m < M) & (x_k < K),
            other=0.0)
        x = x
        w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
        w = tl.load(w_ptr + w_k * N + offs_n, mask=(w_k < K) & (offs_n < N),
            other=0.0)
        w = w
        z = tl.dot(x, w, acc=z)
    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        z += b
    z_offset = offs_m * N + offs_n
    z_mask = (offs_m < M) & (offs_n < N)
    if apply_gelu:
        z = gelu_new(z)
    if dropout_prob > 0.0:
        z = dropout(z, dropout_prob, seed, z_offset)
    if r_ptr is not None:
        r = tl.load(r_ptr + z_offset, mask=z_mask)
        z += r
    tl.store(z_ptr + z_offset, z, mask=z_mask)
