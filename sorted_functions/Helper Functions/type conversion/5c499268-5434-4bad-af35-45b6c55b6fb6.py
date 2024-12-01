import triton
import triton.language as tl
import torch

@triton.jit
def rotary_kernel(OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS, seqlen,
    nheads, rotary_dim, seqlen_ro, CACHE_KEY_SEQLEN, stride_out_batch,
    stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    BLOCK_K: 'tl.constexpr', IS_SEQLEN_OFFSETS_TENSOR: 'tl.constexpr',
    IS_VARLEN: 'tl.constexpr', INTERLEAVED: 'tl.constexpr', CONJUGATE:
    'tl.constexpr', BLOCK_M: 'tl.constexpr'):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2
    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = (OUT + start_idx * stride_out_seqlen + pid_head *
            stride_out_nheads)
    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)
    if not INTERLEAVED:
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] *
            stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[
            None, :] < rotary_dim_half), other=1.0)
        sin = tl.load(SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[
            None, :] < rotary_dim_half), other=0.0)
        x0 = tl.load(X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] <
            rotary_dim_half), other=0.0)
        x1 = tl.load(X + rotary_dim_half * stride_x_headdim, mask=(rm[:,
            None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] *
            stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] <
            rotary_dim_half))
        tl.store(OUT + rotary_dim_half * stride_out_headdim, o1, mask=(rm[:,
            None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    else:
        rk_swap = rk + (rk + 1) % 2 * 2 - 1
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] *
            stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] *
            stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[
            None, :] < rotary_dim_half), other=1.0)
        sin = tl.load(SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[
            None, :] < rotary_dim_half), other=0.0)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] <
            rotary_dim), other=0.0)
        x1 = tl.load(X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] <
            rotary_dim), other=0.0)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] *
            stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] <
            rotary_dim))
