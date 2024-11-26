import triton
import triton.language as tl
import torch

@triton.autotune(configs=get_cuda_autotune_config(), key=['N_ITER', 'M',
    'N', 'K'], reset_to_zero=['output_ptr'])
@triton.jit
def fused_unpack_and_reconstruct_kernel_v3(packed_sign_ptr, u_ptr, vt_ptr,
    output_ptr, N_ITER, M, N, K, n_sign_elements, stride_packed_sign_iter,
    stride_packed_sign_m, stride_packed_sign_n, stride_u_iter, stride_u_m,
    stride_u_k, stride_vt_iter, stride_vt_k, stride_vt_n,
    stride_output_iter, stride_output_m, stride_output_n, BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K:
    'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr', num_warps: 'tl.constexpr',
    num_stages: 'tl.constexpr'):
    pid_spatial = tl.program_id(axis=0)
    pid_iter = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_spatial // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    local_pid = pid_spatial % num_pid_in_group
    pid_n = local_pid // group_size_m
    pid_m = first_pid_m + local_pid % group_size_m
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    PACKED_BLOCK_SIZE_N: 'tl.constexpr' = BLOCK_SIZE_N // 8
    offsets_n_packed = pid_n * PACKED_BLOCK_SIZE_N + tl.arange(0,
        PACKED_BLOCK_SIZE_N)
    packed_sign_ptrs = packed_sign_ptr + (pid_iter *
        stride_packed_sign_iter + offsets_m[:, None] * stride_packed_sign_m +
        offsets_n_packed[None, :] * stride_packed_sign_n)
    packed_bytes = tl.load(packed_sign_ptrs)
    bit_offsets = tl.arange(0, 8)
    packed_bytes = packed_bytes[:, :, None]
    bits = packed_bytes >> 7 - bit_offsets & 1
    signs = bits * 2 - 1
    signs = tl.reshape(signs, (BLOCK_SIZE_M, BLOCK_SIZE_N))
    u_ptrs = u_ptr + pid_iter * stride_u_iter + offsets_m[:, None] * stride_u_m
    vt_ptrs = vt_ptr + pid_iter * stride_vt_iter + offsets_n[None, :
        ] * stride_vt_n
    iter_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        u_block_ptrs = u_ptrs + offsets_k[None, :] * stride_u_k
        vt_block_ptrs = vt_ptrs + offsets_k[:, None] * stride_vt_k
        u = tl.load(u_block_ptrs, mask=offsets_k[None, :] < K - k *
            BLOCK_SIZE_K, other=0.0)
        vt = tl.load(vt_block_ptrs, mask=offsets_k[:, None] < K - k *
            BLOCK_SIZE_K, other=0.0)
        iter_acc += tl.dot(u, vt, out_dtype=tl.float32)
    output = signs * iter_acc
    offsets_output_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_output_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (output_ptr + pid_iter * stride_output_iter + 
        stride_output_m * offsets_output_m[:, None] + stride_output_n *
        offsets_output_n[None, :])
    output_mask = (offsets_output_m[:, None] < M) & (offsets_output_n[None,
        :] < N)
    tl.atomic_add(output_ptrs, output, mask=output_mask)
