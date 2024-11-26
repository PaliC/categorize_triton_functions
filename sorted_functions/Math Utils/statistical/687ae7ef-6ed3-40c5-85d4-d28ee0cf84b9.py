import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_H': 1}), triton.Config
    ({'BLOCK_SIZE_H': 2}), triton.Config({'BLOCK_SIZE_H': 4}), triton.
    Config({'BLOCK_SIZE_H': 8}), triton.Config({'BLOCK_SIZE_H': 16}),
    triton.Config({'BLOCK_SIZE_H': 32}), triton.Config({'BLOCK_SIZE_H': 64}
    )], key=['chunk_size', 'nheads'])
@triton.jit
def _chunk_cumsum_fwd_kernel(dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr,
    dA_cumsum_ptr, batch, seqlen, nheads, chunk_size, dt_min, dt_max,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head, stride_A_head,
    stride_dt_bias_head, stride_dt_out_batch, stride_dt_out_chunk,
    stride_dt_out_head, stride_dt_out_csize, stride_dA_cs_batch,
    stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, DT_SOFTPLUS:
    'tl.constexpr', HAS_DT_BIAS: 'tl.constexpr', BLOCK_SIZE_H:
    'tl.constexpr', BLOCK_SIZE_CHUNK: 'tl.constexpr'):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] *
        stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + 
        offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + 
        offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :
        ] < chunk_size_limit), other=0.0)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=
            offs_h < nheads, other=0.0)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] <
        chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[
        None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[
        None, :] < chunk_size))
