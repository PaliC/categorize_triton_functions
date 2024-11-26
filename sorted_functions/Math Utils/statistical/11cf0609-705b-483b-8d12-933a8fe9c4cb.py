import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_H': 1}, pre_hook=
    init_to_zero(['dA_ptr', 'ddt_bias_ptr'])), triton.Config({
    'BLOCK_SIZE_H': 2}, pre_hook=init_to_zero(['dA_ptr', 'ddt_bias_ptr'])),
    triton.Config({'BLOCK_SIZE_H': 4}, pre_hook=init_to_zero(['dA_ptr',
    'ddt_bias_ptr'])), triton.Config({'BLOCK_SIZE_H': 8}, pre_hook=
    init_to_zero(['dA_ptr', 'ddt_bias_ptr'])), triton.Config({
    'BLOCK_SIZE_H': 16}, pre_hook=init_to_zero(['dA_ptr', 'ddt_bias_ptr'])),
    triton.Config({'BLOCK_SIZE_H': 32}, pre_hook=init_to_zero(['dA_ptr',
    'ddt_bias_ptr'])), triton.Config({'BLOCK_SIZE_H': 64}, pre_hook=
    init_to_zero(['dA_ptr', 'ddt_bias_ptr']))], key=['chunk_size', 'nheads'])
@triton.jit
def _chunk_cumsum_bwd_kernel(ddA_ptr, ddt_out_ptr, dt_ptr, A_ptr,
    dt_bias_ptr, ddt_ptr, dA_ptr, ddt_bias_ptr, batch, seqlen, nheads,
    chunk_size, dt_min, dt_max, stride_ddA_batch, stride_ddA_chunk,
    stride_ddA_head, stride_ddA_csize, stride_ddt_out_batch,
    stride_ddt_out_chunk, stride_ddt_out_head, stride_ddt_out_csize,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head, stride_A_head,
    stride_dt_bias_head, stride_ddt_batch, stride_ddt_seqlen,
    stride_ddt_head, stride_dA_head, stride_ddt_bias_head, DT_SOFTPLUS:
    'tl.constexpr', HAS_DT_BIAS: 'tl.constexpr', BLOCK_SIZE_H:
    'tl.constexpr', BLOCK_SIZE_CHUNK: 'tl.constexpr'):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    ddt_out_ptr += pid_b * stride_ddt_out_batch + pid_c * stride_ddt_out_chunk
    ddA_ptr += pid_b * stride_ddA_batch + pid_c * stride_ddA_chunk
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    ddt_ptr += (pid_b * stride_ddt_batch + pid_c * chunk_size *
        stride_ddt_seqlen)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    ddt_out_ptrs = ddt_out_ptr + (offs_h[:, None] * stride_ddt_out_head + 
        offs_c[None, :] * stride_ddt_out_csize)
    ddA_ptrs = ddA_ptr + (offs_h[:, None] * stride_ddA_head + offs_c[None,
        :] * stride_ddA_csize)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] *
        stride_dt_seqlen)
    ddt_ptrs = ddt_ptr + (offs_h[:, None] * stride_ddt_head + offs_c[None,
        :] * stride_ddt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    ddA = tl.load(ddA_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None,
        :] < chunk_size_limit), other=0.0)
    ddt_out = tl.load(ddt_out_ptrs, mask=(offs_h[:, None] < nheads) & (
        offs_c[None, :] < chunk_size_limit), other=0.0)
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0)
    ddt = ddA * A[:, None] + ddt_out
    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :
        ] < chunk_size_limit), other=0.0)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=
            offs_h < nheads, other=0.0)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt_presoftplus = dt
        dt = softplus(dt)
    clamp_mask = (dt < dt_min) | (dt > dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] <
        chunk_size_limit), dt, 0.0)
    ddt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] <
        chunk_size_limit), ddt, 0.0)
    ddt = tl.where(clamp_mask, 0.0, ddt)
    if DT_SOFTPLUS:
        ddt = tl.where(dt_presoftplus <= 20.0, ddt * tl.sigmoid(
            dt_presoftplus), ddt)
    tl.store(ddt_ptrs, ddt, mask=(offs_h[:, None] < nheads) & (offs_c[None,
        :] < chunk_size_limit))
    dA = tl.sum(ddA * dt, axis=1)
    tl.atomic_add(dA_ptr + offs_h * stride_dA_head, dA, mask=offs_h < nheads)
    if HAS_DT_BIAS:
        ddt_bias = tl.sum(ddt, axis=1)
        tl.atomic_add(ddt_bias_ptr + offs_h * stride_ddt_bias_head,
            ddt_bias, mask=offs_h < nheads)
