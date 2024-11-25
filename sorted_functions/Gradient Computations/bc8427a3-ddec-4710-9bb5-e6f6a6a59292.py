import triton
import triton.language as tl
import torch

@triton.jit
def _splitK_reduce_varargs_backward(Out_splitK: "'VAR_ARGS_ARRAY'",
    LSE_splitK: "'VAR_ARGS_ARRAY'", Dout_splitK: "'VAR_ARGS_ARRAY'",
    DLSE_splitK: "'VAR_ARGS_ARRAY'", Out, LSE, DOut, DLSE, stride_osk_z:
    "'VAR_ARGS_ARRAY'", stride_osk_g: "'VAR_ARGS_ARRAY'", stride_osk_h:
    "'VAR_ARGS_ARRAY'", stride_osk_m: "'VAR_ARGS_ARRAY'", stride_osk_k:
    "'VAR_ARGS_ARRAY'", stride_lsek_z: "'VAR_ARGS_ARRAY'", stride_lsek_g:
    "'VAR_ARGS_ARRAY'", stride_lsek_h: "'VAR_ARGS_ARRAY'", stride_lsek_m:
    "'VAR_ARGS_ARRAY'", stride_oz, stride_og, stride_oh, stride_om,
    stride_ok, stride_lse_z, stride_lse_g, stride_lse_h, stride_lse_m,
    stride_doz, stride_dog, stride_doh, stride_dom, stride_dok,
    stride_dlse_z, stride_dlse_g, stride_dlse_h, stride_dlse_m, BLOCK_SIZE:
    'tl.constexpr', H: 'tl.constexpr', G: 'tl.constexpr'):
    """
    Backward for _splitK_reduce_varargs. Similar to forward, it takes
    attention and LSE of chunks as lists of tensors,
    and outputs the corresponding gradients in the same format.
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
    offset_out = (stride_oz * off_z + stride_oh * off_h + stride_og * off_g +
        stride_om * off_m + tl.arange(0, BLOCK_SIZE))
    offset_dout = (stride_doz * off_z + stride_doh * off_h + stride_dog *
        off_g + stride_dom * off_m + tl.arange(0, BLOCK_SIZE))
    out = tl.load(Out + offset_out)
    dattn = tl.load(DOut + offset_dout)
    offset_lse = (stride_lse_z * off_z + stride_lse_h * off_h + 
        stride_lse_g * off_g + stride_lse_m * off_m)
    offset_dlse = (stride_dlse_z * off_z + stride_dlse_h * off_h + 
        stride_dlse_g * off_g + stride_dlse_m * off_m)
    lse = tl.load(LSE + offset_lse)
    dlse = tl.load(DLSE + offset_dlse)
    for split_k_idx in range(len(Out_splitK)):
        out_splitk = tl.load(Out_splitK[split_k_idx] + out_splitk_offset[
            split_k_idx])
        lse_splitk = tl.load(LSE_splitK[split_k_idx] + lse_splitk_offset[
            split_k_idx])
        dout_splitk_ptr = Dout_splitK[split_k_idx] + out_splitk_offset[
            split_k_idx]
        dlse_splitk_ptr = DLSE_splitK[split_k_idx] + lse_splitk_offset[
            split_k_idx]
        dattn_dattn_i = tl.exp(lse_splitk - lse_max) / tl.exp(lse - lse_max)
        dX_dattn_i = dattn_dattn_i * dattn
        tl.store(dout_splitk_ptr, dX_dattn_i)
        dattn_dlse_i = (out_splitk - out) * dattn_dattn_i
        dlse_dlse_i = dattn_dattn_i
        dX_dlse_i = dlse_dlse_i * dlse + tl.sum(dattn_dlse_i * dattn)
        tl.store(dlse_splitk_ptr, dX_dlse_i)
