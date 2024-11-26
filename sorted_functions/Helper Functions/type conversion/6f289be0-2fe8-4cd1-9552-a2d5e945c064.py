import triton
import triton.language as tl
import torch

@triton.jit
def _splitK_reduce(Out_splitK, LSE_splitK, Out, LSE, split_k:
    'tl.constexpr', splitK_pow2: 'tl.constexpr', stride_osk_z:
    'tl.constexpr', stride_osk_g: 'tl.constexpr', stride_osk_h:
    'tl.constexpr', stride_osk_s: 'tl.constexpr', stride_osk_m:
    'tl.constexpr', stride_osk_k: 'tl.constexpr', stride_lsek_z:
    'tl.constexpr', stride_lsek_g: 'tl.constexpr', stride_lsek_h:
    'tl.constexpr', stride_lsek_s: 'tl.constexpr', stride_lsek_m:
    'tl.constexpr', stride_oz: 'tl.constexpr', stride_og: 'tl.constexpr',
    stride_oh: 'tl.constexpr', stride_om: 'tl.constexpr', stride_ok:
    'tl.constexpr', stride_lse_z: 'tl.constexpr', stride_lse_g:
    'tl.constexpr', stride_lse_h: 'tl.constexpr', stride_lse_m:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', H: 'tl.constexpr', G:
    'tl.constexpr', WRITE_LSE: 'tl.constexpr'):
    off_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = off_zhg // G % H
    off_g = off_zhg % G
    Out_splitK_ptr = (Out_splitK + stride_osk_z * off_z + stride_osk_g *
        off_g + stride_osk_h * off_h + stride_osk_m * off_m + tl.arange(0,
        BLOCK_SIZE)[None, :] + stride_osk_s * tl.arange(0, splitK_pow2)[:,
        None])
    LSE_splitK_ptr0 = (LSE_splitK + stride_lsek_z * off_z + stride_lsek_g *
        off_g + stride_lsek_h * off_h + stride_lsek_m * off_m + 
        stride_lsek_s * tl.arange(0, splitK_pow2))
    if splitK_pow2 > split_k:
        mask_1d = tl.arange(0, splitK_pow2) < split_k
        mask_2d = mask_1d[:, None]
        lse_splitk = tl.load(LSE_splitK_ptr0, mask=mask_1d, other=float('-inf')
            )
        lse_max = tl.max(lse_splitk)
        out_splitk = tl.load(Out_splitK_ptr, mask=mask_2d, other=0)
        lse_splitk = tl.load(LSE_splitK_ptr0, mask=mask_1d, other=float('-inf')
            )
    else:
        lse_splitk = tl.load(LSE_splitK_ptr0)
        lse_max = tl.max(lse_splitk)
        out_splitk = tl.load(Out_splitK_ptr)
        lse_splitk = tl.load(LSE_splitK_ptr0)
    sumexp_normalized_splitk = tl.math.exp2((lse_splitk - lse_max) * 1.44269504
        )
    sumexp_normalized = tl.sum(sumexp_normalized_splitk, axis=0)
    numerator_normalized = tl.sum(out_splitk * sumexp_normalized_splitk[:,
        None], axis=0)
    acc = numerator_normalized / sumexp_normalized
    acc = tl.where(lse_max == float('-inf'), 0.0, acc)
    Out_ptr = (Out + stride_oz * off_z + stride_oh * off_h + stride_og *
        off_g + stride_om * off_m + tl.arange(0, BLOCK_SIZE))
    if acc.dtype is tl.float64 and Out.dtype.element_ty is not tl.float64:
        acc = acc
    tl.store(Out_ptr, acc)
    if WRITE_LSE:
        l_ptrs = (LSE + off_z * stride_lse_z + off_g * stride_lse_g + off_h *
            stride_lse_h + off_m * stride_lse_m)
        to_store = lse_max + tl.math.log2(sumexp_normalized) / 1.44269504
        to_store = tl.where(lse_max == float('-inf'), lse_max, to_store)
        tl.store(l_ptrs, to_store)
