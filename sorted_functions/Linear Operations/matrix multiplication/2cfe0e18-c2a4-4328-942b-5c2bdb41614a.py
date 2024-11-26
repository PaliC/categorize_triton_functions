import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N':
    32, 'BLOCK_SIZE_K': 16}, num_stages=2, num_warps=4), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16}, num_stages
    =2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 
    64, 'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 16},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 16}, num_stages=4, num_warps=8)],
    key=['N_ITER', 'M', 'N', 'K'])
@triton.jit
def fused_reconstruct_and_forward_kernel(sign_ptr, u_ptr, vt_ptr,
    output_ptr, x_ptr, N_ITER, M, N, K, BZ, L, stride_sign_iter,
    stride_sign_m, stride_sign_n, stride_u_iter, stride_u_m, stride_u_k,
    stride_vt_iter, stride_vt_k, stride_vt_n, stride_x_bz, stride_x_l,
    stride_x_n, stride_output_bz, stride_output_l, stride_output_m,
    BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr',
    BLOCK_SIZE_K: 'tl.constexpr', num_warps: 'tl.constexpr', num_stages:
    'tl.constexpr'):
    """Kernel for computing (sign * u @ vt).sum(dim=0) followed by x @ w.T.
    sign: (N_ITER, M, N)
    u: (N_ITER, M, K)
    vt: (N_ITER, K, N)
    x: (BZ, L, N)
    output: (BZ, L, M)
    """
    pid_m = tl.program_id(axis=0)
    pid_bzl = tl.program_id(axis=1)
    bz = pid_bzl // L
    l = pid_bzl % L
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    output_vals = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for n_off in range(0, N, BLOCK_SIZE_N):
        offsets_n = n_off + tl.arange(0, BLOCK_SIZE_N)
        n_mask = offsets_n < N
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
        for n_iter in range(N_ITER):
            sign_ptrs = sign_ptr + (n_iter * stride_sign_iter + offsets_m[:,
                None] * stride_sign_m + offsets_n[None, :] * stride_sign_n)
            u_ptrs = u_ptr + (n_iter * stride_u_iter + offsets_m[:, None] *
                stride_u_m)
            vt_ptrs = vt_ptr + (n_iter * stride_vt_iter + offsets_n[None, :
                ] * stride_vt_n)
            iter_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
            for k in range(0, K, BLOCK_SIZE_K):
                u_block_ptrs = u_ptrs + (offsets_k[None, :] + k) * stride_u_k
                vt_block_ptrs = vt_ptrs + (offsets_k[:, None] + k
                    ) * stride_vt_k
                k_mask = offsets_k + k < K
                u = tl.load(u_block_ptrs, mask=k_mask[None, :], other=0.0)
                vt = tl.load(vt_block_ptrs, mask=k_mask[:, None], other=0.0)
                iter_acc += tl.dot(u, vt, out_dtype=tl.float16)
            sign = tl.load(sign_ptrs, mask=(offsets_m[:, None] < M) &
                n_mask[None, :], other=0.0)
            acc += sign * iter_acc
        x_ptrs = (x_ptr + bz * stride_x_bz + l * stride_x_l + offsets_n *
            stride_x_n)
        x_vals = tl.load(x_ptrs, mask=n_mask, other=0.0)
        output_vals += tl.sum(acc * x_vals[None, :], axis=1)
    output_ptrs = (output_ptr + bz * stride_output_bz + l * stride_output_l +
        offsets_m * stride_output_m)
    tl.store(output_ptrs, output_vals, mask=offsets_m < M)
