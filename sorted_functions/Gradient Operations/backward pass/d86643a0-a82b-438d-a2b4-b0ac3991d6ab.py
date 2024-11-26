import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32}), triton.
    Config({'BLOCK_SIZE_M': 64}), triton.Config({'BLOCK_SIZE_M': 128}),
    triton.Config({'BLOCK_SIZE_M': 256})], key=['chunk_size', 'hdim'])
@triton.jit
def _chunk_scan_bwd_ddAcs_unstable_kernel(dout_ptr, out_ptr, dt_ptr,
    ddt_ptr, x_ptr, D_ptr, ddA_cumsum_ptr, dD_ptr, chunk_size, hdim, batch,
    seqlen, stride_dout_batch, stride_dout_seqlen, stride_dout_head,
    stride_dout_hdim, stride_out_batch, stride_out_seqlen, stride_out_head,
    stride_out_hdim, stride_dt_batch, stride_dt_chunk, stride_dt_head,
    stride_dt_csize, stride_ddt_batch, stride_ddt_chunk, stride_ddt_head,
    stride_ddt_csize, stride_x_batch, stride_x_seqlen, stride_x_head,
    stride_x_hdim, stride_D_head, stride_ddA_cs_batch, stride_ddA_cs_chunk,
    stride_ddA_cs_head, stride_ddA_cs_csize, stride_dD_batch,
    stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim, HAS_D:
    'tl.constexpr', D_HAS_HDIM: 'tl.constexpr', SUBTRACT_DDTDT:
    'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'
    ):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    dout_ptr += (pid_b * stride_dout_batch + pid_c * chunk_size *
        stride_dout_seqlen + pid_h * stride_dout_head)
    out_ptr += (pid_b * stride_out_batch + pid_c * chunk_size *
        stride_out_seqlen + pid_h * stride_out_head)
    dt_ptr += (pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h *
        stride_dt_head)
    ddt_ptr += (pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h *
        stride_ddt_head)
    ddA_cumsum_ptr += (pid_b * stride_ddA_cs_batch + pid_c *
        stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head)
    if HAS_D:
        x_ptr += (pid_b * stride_x_batch + pid_c * chunk_size *
            stride_x_seqlen + pid_h * stride_x_head)
        dD_ptr += (pid_b * stride_dD_batch + pid_c * stride_dD_chunk + 
            pid_h * stride_dD_head + pid_m * stride_dD_csize)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[
        None, :] * stride_dout_hdim)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None,
        :] * stride_out_hdim)
    if HAS_D:
        x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None,
            :] * stride_x_hdim)
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < hdim), other=0.0)
    out = tl.load(out_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < hdim), other=0.0)
    if HAS_D:
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_n[None, :] < hdim), other=0.0)
        if D_HAS_HDIM:
            dD = tl.sum(dout * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n <
                hdim, other=0.0)
        else:
            dD = tl.sum(dout * x)
            tl.store(dD_ptr, dD)
            D = tl.load(D_ptr + pid_h * stride_D_head)
        out -= x * D
    ddA_cs = tl.sum(dout * out, axis=1)
    if SUBTRACT_DDTDT:
        dt = tl.load(dt_ptr + offs_m * stride_dt_csize, mask=offs_m <
            chunk_size, other=0.0)
        ddt = tl.load(ddt_ptr + offs_m * stride_ddt_csize, mask=offs_m <
            chunk_size, other=0.0)
        ddA_cs -= dt * ddt
    tl.store(ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize, ddA_cs, mask=
        offs_m < chunk_size)
