import triton
import triton.language as tl
import torch

@triton.jit
def _mixed_mm_kernel(A, B, scales_ptr, zeros_ptr, C, M, N, K, stride_am,
    stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scale_k,
    stride_scale_n, IS_BFLOAT16: 'tl.constexpr', QGROUP_SIZE:
    'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    BLOCK_K: 'tl.constexpr', SPLIT_K: 'tl.constexpr', EVEN_K:
    'tl.constexpr', TRANSPOSED: 'tl.constexpr'=False, GROUP_M:
    'tl.constexpr'=8, acc_dtype: 'tl.constexpr'=tl.float32, input_precision:
    'tl.constexpr'='ieee', fp8_fast_accum: 'tl.constexpr'=False, DEBUG:
    'tl.constexpr'=False):
    """Mixed matmul kernel

    A has shape (M, K) and is float16, bfloat16, or float32

    B is i4 / s4 and has shape (K // 2, N) and is packed as uint8 / int8. See `packed_2xint4` for details.

    Scales and zeros are of shape (NUM_GROUPS, N) and are same dtype as A, where NUM_GROUPS = (K // QGROUP_SIZE)
    QGROUP_SIZE should be a multiple of BLOCK_K such that a vector of scales / zeros is loaded and broadcasted to block shape
    per mainloop iteration.

    In the transposed case, A is M x N and B is K x N, and we reduce along "N":
    - TLDR: we are loading rows of A and B blocks at a time, dequantizing and transposing each block of B to achieve the overall
    effect of a transposed matmul. This is necessary to perform a transposed matmul without unpacking and repacking the B matrix.
        - Indexing remains the same for A (the reduction dim (BLK_K / K) corresponds to axis 1 of A -- "N" above)
            - We load a BLK_M x BLK_K block of A
        - Indexing for B is now flipped: N <-> K
            - We load BLK_N x BLK_K block of B (remembering that the reduction dimension is axis 1 of B)
            - We dequantize and transpose to BLK_K x BLK_N
            - scale / zero indexing also change, since we are now iterating along the non-grouping dim within the mac loop and along
            the grouping dim across blocks.
        - Each mac loop calculates BLK_M x BLK_N -> M x "N"(= K)
        - Within the mac loop for each block, we iterate along axis=1 for **both** A and B since axis = 1 is now the reduction dim for B.

    NOTE: Assumes that the quantization grouping was done along the K dimension originally (i.e., QGROUP_SIZE consecutive elements
    of original weight matrix in the K dimension were grouped together when calculating min / max scaling factors).
    """
    if not TRANSPOSED:
        tl.static_assert(QGROUP_SIZE % BLOCK_K == 0)
    else:
        tl.static_assert(QGROUP_SIZE % BLOCK_N == 0)
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    if not DEBUG:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm
    rak = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    if not TRANSPOSED:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        if not DEBUG:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        else:
            rbn = rn
        rbk = pid_z * BLOCK_K // 2 + tl.arange(0, BLOCK_K // 2)
    else:
        rn = (pid_n * BLOCK_N // 2 + tl.arange(0, BLOCK_N // 2)) % N
        if not DEBUG:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N // 2), 
                BLOCK_N // 2)
        else:
            rbn = rn
        rbk = rak
    A = A + (ram[:, None] * stride_am + rak[None, :] * stride_ak)
    if not TRANSPOSED:
        B = B + (rbk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    else:
        B = B + (rbn[:, None] * stride_bk + rbk[None, :] * stride_bn)
    if not TRANSPOSED:
        offsets_scale_n = pid_n * stride_scale_n * BLOCK_N + tl.arange(0,
            BLOCK_N) * stride_scale_n
    else:
        scale_offset_k = pid_n * BLOCK_N * stride_scale_k // QGROUP_SIZE
        offsets_scale_n = tl.arange(0, BLOCK_K) * stride_scale_n
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            qb = tl.load(B)
        else:
            k_remaining_a = K - k * (BLOCK_K * SPLIT_K)
            if not TRANSPOSED:
                k_remaining_b = K - k * (BLOCK_K * SPLIT_K) // 2
            else:
                k_remaining_b = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rak[None, :] < k_remaining_a, other=_0)
            qb = tl.load(B, mask=rbk[:, None] < k_remaining_b, other=_0)
        if not TRANSPOSED:
            scale_offset_k = (k * BLOCK_K * SPLIT_K * stride_scale_k //
                QGROUP_SIZE)
        else:
            offsets_scale_n = k * stride_scale_n * BLOCK_K + tl.arange(0,
                BLOCK_K) * stride_scale_n
        scales = tl.load(scales_ptr + offsets_scale_n + scale_offset_k)
        zeros = tl.load(zeros_ptr + offsets_scale_n + scale_offset_k)
        _4_i8 = tl.full((1,), 4, dtype=tl.int8)
        qb_lo = qb << _4_i8 >> _4_i8
        qb_hi = qb >> _4_i8
        if IS_BFLOAT16:
            dq_b = tl.join(qb_lo.to(tl.float16), qb_hi.to(tl.float16)).permute(
                0, 2, 1)
        else:
            dq_b = tl.join(qb_lo, qb_hi).permute(0, 2, 1)
        if not TRANSPOSED:
            dq_b = dq_b.reshape(BLOCK_K, BLOCK_N)
        else:
            dq_b = dq_b.reshape(BLOCK_N, BLOCK_K)
        zeros = zeros[None, :]
        scales = scales[None, :]
        dq_b = (dq_b - zeros) * scales
        if TRANSPOSED:
            dq_b = tl.trans(dq_b)
        if fp8_fast_accum:
            acc = tl.dot(a, dq_b, acc, out_dtype=acc_dtype, input_precision
                =input_precision)
        else:
            acc += tl.dot(a, dq_b, out_dtype=acc_dtype, input_precision=
                input_precision)
        A += BLOCK_K * SPLIT_K * stride_ak
        if not TRANSPOSED:
            B += BLOCK_K * SPLIT_K * stride_bk // 2
        else:
            B += BLOCK_K * SPLIT_K * stride_bn
    acc = acc
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)
