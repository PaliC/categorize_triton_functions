import triton
import triton.language as tl
import torch

@autotune(get_small_k_configs(), key=['M', 'N', 'K'], prune_configs_by={
    'early_config_prune': small_k_early_config_prune, 'perf_model':
    estimate_matmul_time, 'top_k': _AUTOTUNE_TOPK})
@triton.jit
def _mm_small_k_kernel(A, B, M, N, K, stride_am, stride_ak, stride_bk,
    stride_bn, acc_dtype: 'tl.constexpr', input_precision: 'tl.constexpr',
    fp8_fast_accum: 'tl.constexpr', BLOCK_K: 'tl.constexpr', AB_DTYPE:
    'tl.constexpr', BLOCK_M: 'tl.constexpr'=256, BLOCK_N: 'tl.constexpr'=64,
    C=None, stride_cm=None, stride_cn=None, Norm2=None, Source=None,
    stride_sourcem=None, stride_sourcen=None, Magnitude=None, ADD_SOURCE:
    'tl.constexpr'=False, EPILOGUE_NORM: 'tl.constexpr'=False,
    EPILOGUE_MAGNITUDE: 'tl.constexpr'=False, STORE_ACC: 'tl.constexpr'=False):
    pid_m = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    a = tl.load(A)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    rn = tl.arange(0, BLOCK_N)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    if STORE_ACC:
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    if ADD_SOURCE:
        Source = Source + (rm[:, None] * stride_sourcem + rn[None, :] *
            stride_sourcen)
    if EPILOGUE_NORM:
        norm_vec = tl.zeros((BLOCK_M,), dtype=acc_dtype)
    if EPILOGUE_MAGNITUDE:
        Magnitude = Magnitude + ram
    mask_m = rm < M
    for n in range(0, tl.cdiv(N, BLOCK_N)):
        b = tl.load(B)
        if AB_DTYPE is not None:
            a = a
            b = b
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, input_precision=
                input_precision)
        else:
            acc = tl.dot(a, b, out_dtype=acc_dtype, input_precision=
                input_precision)
        if ADD_SOURCE:
            mask_n = (n * BLOCK_N + rn < N)[None, :]
            source = tl.load(Source, mask=mask_m[:, None] & mask_n)
            acc += source
            Source += BLOCK_N * stride_sourcen
        if EPILOGUE_NORM:
            norm_vec += tl.sum(acc * acc, axis=1)
        if STORE_ACC:
            mask_n = (n * BLOCK_N + rn < N)[None, :]
            tl.store(C, acc, mask=mask_m[:, None] & mask_n)
            C += BLOCK_N * stride_cn
        B += BLOCK_N * stride_bn
    if EPILOGUE_NORM:
        Norm2 = Norm2 + rm
        norm_vec = tl.rsqrt(norm_vec)
        if EPILOGUE_MAGNITUDE:
            magnitude = tl.load(Magnitude, mask=mask_m)
            norm_vec *= magnitude
        tl.store(Norm2, norm_vec, mask=mask_m)
