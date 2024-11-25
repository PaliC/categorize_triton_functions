import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_combine_kv_splits(multiple_o, multiple_l, final_o, final_l,
    stride_mul_oz, stride_mul_oh, stride_mul_os, stride_mul_om,
    stride_mul_ok, stride_fin_oz, stride_fin_oh, stride_fin_om,
    stride_fin_ok, Z, H, M, S, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL:
    'tl.constexpr', DIVISIBLE_M: 'tl.constexpr'):
    start_m = tl.program_id(0)
    offs_h = tl.program_id(1)
    offs_z = tl.program_id(2)
    multiple_o += offs_z * stride_mul_oz + offs_h * stride_mul_oh
    multiple_l += (offs_z * H + offs_h) * S * M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not DIVISIBLE_M:
        mask_m = offs_m < M
    m = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
    acc = tl.full([BLOCK_M], value=float(0.0), dtype=tl.float32)
    l_ptrs = multiple_l + offs_m
    for _ in range(0, S):
        if DIVISIBLE_M:
            l = tl.load(l_ptrs)
        else:
            l = tl.load(l_ptrs, mask=mask_m)
        m_new = tl.maximum(m, l)
        acc = acc * tl.exp(m - m_new) + tl.exp(l - m_new)
        m = m_new
        l_ptrs += M
    l_acc = m + tl.log(acc)
    o_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_ptrs = multiple_l + offs_m
    offs_k = tl.arange(0, BLOCK_DMODEL)
    o_ptrs = multiple_o + offs_m[:, None] * stride_mul_om + offs_k[None, :
        ] * stride_mul_ok
    for _ in range(0, S):
        l = tl.load(l_ptrs, mask=offs_m < M)
        rescale = tl.exp(l - l_acc)
        if DIVISIBLE_M:
            o = tl.load(o_ptrs)
        else:
            o = tl.load(o_ptrs, mask=mask_m[:, None])
        o_acc += o * rescale[:, None]
        l_ptrs += M
        o_ptrs += stride_mul_os
    final_o += offs_z * stride_fin_oz + offs_h * stride_fin_oh
    final_l += (offs_z * H + offs_h) * M
    a_ptrs = final_o + offs_m[:, None] * stride_fin_om + offs_k * stride_fin_ok
    b_ptrs = final_l + offs_m
    if DIVISIBLE_M:
        tl.store(a_ptrs, o_acc)
        tl.store(b_ptrs, l_acc)
    else:
        tl.store(a_ptrs, o_acc, mask=mask_m[:, None])
        tl.store(b_ptrs, l_acc, mask=mask_m)
