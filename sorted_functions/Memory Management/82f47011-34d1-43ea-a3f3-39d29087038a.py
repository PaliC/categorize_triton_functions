import triton
import triton.language as tl
import torch

@triton.jit
def _splitK_reduce_varargs(Out_splitK: "'VAR_ARGS_ARRAY'", LSE_splitK:
    "'VAR_ARGS_ARRAY'", Out, LSE, stride_osk_z: "'VAR_ARGS_ARRAY'",
    stride_osk_g: "'VAR_ARGS_ARRAY'", stride_osk_h: "'VAR_ARGS_ARRAY'",
    stride_osk_m: "'VAR_ARGS_ARRAY'", stride_osk_k: "'VAR_ARGS_ARRAY'",
    stride_lsek_z: "'VAR_ARGS_ARRAY'", stride_lsek_g: "'VAR_ARGS_ARRAY'",
    stride_lsek_h: "'VAR_ARGS_ARRAY'", stride_lsek_m: "'VAR_ARGS_ARRAY'",
    stride_oz, stride_og, stride_oh, stride_om, stride_ok, stride_lse_z,
    stride_lse_g, stride_lse_h, stride_lse_m, BLOCK_SIZE: 'tl.constexpr', H:
    'tl.constexpr', G: 'tl.constexpr', WRITE_LSE: 'tl.constexpr'):
    """
    This version of reduce kernel takes attention and LSE of chunks as lists of tensors,
    as opposed to _splitK_reduce, which takes each as a stacked tensor.
    """
    off_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = off_zhg // G % H
    off_g = off_zhg % G
    out_splitk_offset: "'VAR_ARGS_ARRAY'"
    for i in range(len(Out_splitK)):
        out_splitk_offset[i] = stride_osk_z[i] * off_z + stride_osk_g[i
            ] * off_g + stride_osk_h[i] * off_h + stride_osk_m[i
            ] * off_m + tl.arange(0, BLOCK_SIZE)
    lse_splitk_offset: "'VAR_ARGS_ARRAY'"
    for i in range(len(Out_splitK)):
        lse_splitk_offset[i] = stride_lsek_z[i] * off_z + stride_lsek_g[i
            ] * off_g + stride_lsek_h[i] * off_h + stride_lsek_m[i] * off_m
    lse_max = float('-inf')
    for split_k_idx in range(len(Out_splitK)):
        LSE_splitK_ptr = LSE_splitK[split_k_idx] + lse_splitk_offset[
            split_k_idx]
        lse_splitk = tl.load(LSE_splitK_ptr)
        lse_max = tl.maximum(lse_max, lse_splitk)
    sumexp_normalized = 0.0
    numerator_normalized = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for split_k_idx in range(len(Out_splitK)):
        out_splitk = tl.load(Out_splitK[split_k_idx] + out_splitk_offset[
            split_k_idx])
        lse_splitk = tl.load(LSE_splitK[split_k_idx] + lse_splitk_offset[
            split_k_idx])
        sumexp_normalized_splitk = tl.math.exp2((lse_splitk - lse_max) * 
            1.44269504)
        sumexp_normalized += sumexp_normalized_splitk
        numerator_normalized += out_splitk * sumexp_normalized_splitk
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
