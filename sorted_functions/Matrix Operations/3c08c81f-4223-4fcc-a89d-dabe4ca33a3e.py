import triton
import triton.language as tl
import torch

@triton.jit
def _kernel_matmul_fp8_row_tma_persistent(A_ptr, B_ptr, C_ptr, M, N, K,
    A_scale, B_scale, stride_am, stride_ak, stride_bn, stride_bk, stride_cm,
    stride_cn, dot_out_dtype: 'tl.constexpr', allow_tf32: 'tl.constexpr',
    fp8_fast_accum: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', BLOCK_K: 'tl.constexpr', GROUP_M: 'tl.constexpr',
    AB_DTYPE: 'tl.constexpr', NUM_SMS: 'tl.constexpr') ->None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Wether to cast A and B to C.dtype before tensor core.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1
    tile_id = start_pid - NUM_SMS
    ki = -1
    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0
    num_pid_in_group = GROUP_M * num_pid_n
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    dtype_fp8 = tl.float8e4nv
    scale_dtype = tl.float32
    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + tile_id % group_size_m
            pid_n = tile_id % num_pid_in_group // group_size_m
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N
            offs_am = tl.multiple_of(offs_am, BLOCK_M)
            offs_bn = tl.multiple_of(offs_bn, BLOCK_N)
        offs_k = ki * BLOCK_K
        a = tl._experimental_descriptor_load(A_ptr, [offs_am, offs_k], [
            BLOCK_M, BLOCK_K], dtype_fp8)
        b = tl._experimental_descriptor_load(B_ptr, [offs_bn, offs_k], [
            BLOCK_N, BLOCK_K], dtype_fp8)
        acc = tl.dot(a, b.T, acc, out_dtype=dot_out_dtype, allow_tf32=
            allow_tf32)
        if ki == k_tiles - 1:
            rm = pid_m * BLOCK_M
            rn = pid_n * BLOCK_N
            a_scale = tl._experimental_descriptor_load(A_scale, [rm], [
                BLOCK_M], scale_dtype)
            b_scale = tl._experimental_descriptor_load(B_scale, [rn], [
                BLOCK_N], scale_dtype)
            scale = a_scale[:, None] * b_scale[None, :]
            acc *= scale
            acc = acc
            tl._experimental_descriptor_store(C_ptr, acc, [rm, rn])
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
