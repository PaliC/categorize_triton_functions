import triton
import triton.language as tl
import torch

@triton.autotune(configs=get_cuda_autotune_config(), key=['N_ITER', 'M',
    'N', 'K'])
@triton.jit
def fused_unpack_and_reconstruct_kernel(packed_sign_ptr, u_ptr, vt_ptr,
    output_ptr, N_ITER, M, N, K, n_sign_elements, stride_u_iter, stride_u_m,
    stride_u_k, stride_vt_iter, stride_vt_k, stride_vt_n, stride_output_m,
    stride_output_n, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N:
    'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr', GROUP_SIZE_M:
    'tl.constexpr', num_warps: 'tl.constexpr', num_stages: 'tl.constexpr'):
    """Kernel for computing (sign * u @ vt).sum(dim=0) with sign unpacking fused."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    base_m = offsets_m[:, None] * N
    base_n = offsets_n[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for n_iter in range(N_ITER):
        element_indices = n_iter * M * N + base_m + base_n
        element_indices = tl.reshape(element_indices, (BLOCK_SIZE_M *
            BLOCK_SIZE_N,))
        byte_indices = element_indices // 8
        bit_indices = element_indices % 8
        byte_ptrs = packed_sign_ptr + byte_indices
        byte_mask = byte_indices < (n_sign_elements + 7) // 8
        packed_bytes = tl.load(byte_ptrs, mask=byte_mask, other=0)
        bits = packed_bytes >> 7 - bit_indices & 1
        signs = bits * 2 - 1
        signs = tl.reshape(signs, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        u_ptrs = u_ptr + (n_iter * stride_u_iter + offsets_m[:, None] *
            stride_u_m)
        vt_ptrs = vt_ptr + (n_iter * stride_vt_iter + offsets_n[None, :] *
            stride_vt_n)
        iter_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            u_block_ptrs = u_ptrs + offsets_k[None, :] * stride_u_k
            vt_block_ptrs = vt_ptrs + offsets_k[:, None] * stride_vt_k
            u = tl.load(u_block_ptrs, mask=offsets_k[None, :] < K - k *
                BLOCK_SIZE_K, other=0.0)
            vt = tl.load(vt_block_ptrs, mask=offsets_k[:, None] < K - k *
                BLOCK_SIZE_K, other=0.0)
            iter_acc += tl.dot(u, vt, out_dtype=tl.float32)
        acc += signs * iter_acc
    output = acc
    offsets_output_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_output_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + stride_output_m * offsets_output_m[:, None
        ] + stride_output_n * offsets_output_n[None, :]
    output_mask = (offsets_output_m[:, None] < M) & (offsets_output_n[None,
        :] < N)
    tl.store(output_ptrs, output, mask=output_mask)
