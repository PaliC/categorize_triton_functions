import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages
    =3, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages
    =4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages
    =4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=
    4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages
    =4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=
    4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=
    5, num_warps=2), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=
    5, num_warps=2), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=
    4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=
    3, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages
    =2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_stages
    =4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages
    =3, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16},
    num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16},
    num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16},
    num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M':
    32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16},
    num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    num_stages=5, num_warps=2), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    num_stages=5, num_warps=2), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
    num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
    num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16},
    num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16},
    num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16},
    num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16},
    num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16},
    num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M':
    32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16},
    num_stages=2, num_warps=4)], key=['M', 'N', 'K'], reset_to_zero=['c_ptr'])
@triton.jit
def matmul_kernel(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, M, N, K, stride_am,
    stride_ak, stride_asm, stride_bk, stride_bn, stride_bsn, stride_cm,
    stride_cn, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr',
    BLOCK_SIZE_K: 'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr', SPLIT_K:
    'tl.constexpr'):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
        stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] *
        stride_bn)
    as_ptrs = as_ptr + offs_am * stride_asm
    bs_ptrs = bs_ptr + offs_bn * stride_bsn
    a_scale = tl.load(as_ptrs, mask=offs_am < M, other=0.0)
    b_scale = tl.load(bs_ptrs, mask=offs_bn < N, other=0.0)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K *
            SPLIT_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K *
            SPLIT_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    c = accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :
        ]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
