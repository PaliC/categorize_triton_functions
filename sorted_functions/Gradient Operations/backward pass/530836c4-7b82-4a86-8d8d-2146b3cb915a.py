import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32}), triton.
    Config({'BLOCK_SIZE_M': 64}), triton.Config({'BLOCK_SIZE_M': 128}),
    triton.Config({'BLOCK_SIZE_M': 256})], key=['chunk_size', 'hdim'])
@triton.jit
def _chunk_scan_bwd_dz_kernel(dout_ptr, out_ptr, z_ptr, x_ptr, D_ptr,
    outz_ptr, dz_ptr, dout_x_ptr, dD_ptr, ddA_cumsum_ptr, chunk_size, hdim,
    batch, seqlen, stride_dout_batch, stride_dout_seqlen, stride_dout_head,
    stride_dout_hdim, stride_out_batch, stride_out_seqlen, stride_out_head,
    stride_out_hdim, stride_z_batch, stride_z_seqlen, stride_z_head,
    stride_z_hdim, stride_x_batch, stride_x_seqlen, stride_x_head,
    stride_x_hdim, stride_D_head, stride_outz_batch, stride_outz_seqlen,
    stride_outz_head, stride_outz_hdim, stride_dz_batch, stride_dz_seqlen,
    stride_dz_head, stride_dz_hdim, stride_doutx_batch, stride_doutx_seqlen,
    stride_doutx_head, stride_doutx_hdim, stride_dD_batch, stride_dD_chunk,
    stride_dD_head, stride_dD_csize, stride_dD_hdim, stride_ddA_cs_batch,
    stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize, HAS_D:
    'tl.constexpr', D_HAS_HDIM: 'tl.constexpr', HAS_DDACS: 'tl.constexpr',
    RECOMPUTE_OUTPUT: 'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr',
    BLOCK_SIZE_N: 'tl.constexpr'):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    dout_ptr += (pid_b * stride_dout_batch + pid_c * chunk_size *
        stride_dout_seqlen + pid_h * stride_dout_head)
    dout_x_ptr += (pid_b * stride_doutx_batch + pid_c * chunk_size *
        stride_doutx_seqlen + pid_h * stride_doutx_head)
    out_ptr += (pid_b * stride_out_batch + pid_c * chunk_size *
        stride_out_seqlen + pid_h * stride_out_head)
    z_ptr += (pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen +
        pid_h * stride_z_head)
    dz_ptr += (pid_b * stride_dz_batch + pid_c * chunk_size *
        stride_dz_seqlen + pid_h * stride_dz_head)
    if RECOMPUTE_OUTPUT:
        outz_ptr += (pid_b * stride_outz_batch + pid_c * chunk_size *
            stride_outz_seqlen + pid_h * stride_outz_head)
    if HAS_DDACS:
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
    dout_x_ptrs = dout_x_ptr + (offs_m[:, None] * stride_doutx_seqlen + 
        offs_n[None, :] * stride_doutx_hdim)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None,
        :] * stride_out_hdim)
    z_ptrs = z_ptr + (offs_m[:, None] * stride_z_seqlen + offs_n[None, :] *
        stride_z_hdim)
    dz_ptrs = dz_ptr + (offs_m[:, None] * stride_dz_seqlen + offs_n[None, :
        ] * stride_dz_hdim)
    if RECOMPUTE_OUTPUT:
        outz_ptrs = outz_ptr + (offs_m[:, None] * stride_outz_seqlen + 
            offs_n[None, :] * stride_outz_hdim)
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
    z = tl.load(z_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n
        [None, :] < hdim), other=0.0)
    z_sigmoid = tl.sigmoid(z)
    if RECOMPUTE_OUTPUT:
        outz = out * z * z_sigmoid
        tl.store(outz_ptrs, outz, mask=(offs_m[:, None] < chunk_size_limit) &
            (offs_n[None, :] < hdim))
    dz = dout * out * z_sigmoid * (1 + z * (1 - z_sigmoid))
    tl.store(dz_ptrs, dz, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < hdim))
    dout *= z * z_sigmoid
    tl.store(dout_x_ptrs, dout, mask=(offs_m[:, None] < chunk_size_limit) &
        (offs_n[None, :] < hdim))
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
    if HAS_DDACS:
        ddA_cs = tl.sum(dout * out, axis=1)
        tl.store(ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize, ddA_cs,
            mask=offs_m < chunk_size)
