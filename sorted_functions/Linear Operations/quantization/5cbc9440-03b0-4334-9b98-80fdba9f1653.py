import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_stages=2, num_warps=8),
    triton.Config({}, num_stages=2, num_warps=4), triton.Config({},
    num_stages=2, num_warps=2), triton.Config({}, num_stages=2, num_warps=1
    )], key=['K'])
@triton.jit
def quantize_int8_perrow_kernel(fpa_ptr, a_ptr, as_ptr, M, K, stride_fpam,
    stride_fpak, stride_am, stride_ak, stride_asm, BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'):
    pid_m = tl.program_id(axis=0)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    fpa_ptrs = fpa_ptr + offs_am[:, None] * stride_fpam + offs_k[None, :
        ] * stride_fpak
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0)
        a_max = tl.maximum(a_max, tl.max(tl.abs(fpa), axis=1))
        fpa_ptrs += BLOCK_SIZE_K * stride_fpak
    a_scale = a_max / 127.0
    fpa_ptrs = fpa_ptr + offs_am[:, None] * stride_fpam + offs_k[None, :
        ] * stride_fpak
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0)
        inta = fpa / a_scale[:, None]
        tl.store(a_ptrs, inta, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K)
        fpa_ptrs += BLOCK_SIZE_K * stride_fpak
        a_ptrs += BLOCK_SIZE_K * stride_ak
    as_offs = pid_m * BLOCK_SIZE_M * stride_asm + tl.arange(0, BLOCK_SIZE_M)
    tl.store(as_ptr + as_offs, a_scale)
