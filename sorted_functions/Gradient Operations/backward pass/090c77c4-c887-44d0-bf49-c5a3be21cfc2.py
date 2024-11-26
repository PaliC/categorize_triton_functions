import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 16}, num_stages=3,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 32}, num_stages=3,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 64}, num_stages=3,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 128}, num_stages=3,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 16}, num_stages=4,
    num_warps=8), triton.Config({'BLOCK_SIZE_M': 32}, num_stages=4,
    num_warps=8), triton.Config({'BLOCK_SIZE_M': 64}, num_stages=4,
    num_warps=8), triton.Config({'BLOCK_SIZE_M': 128}, num_stages=4,
    num_warps=8)], key=['chunk_size', 'hdim'])
@triton.jit
def _chunk_scan_bwd_ddAcs_stable_kernel_old(x_ptr, dout_ptr, dt_ptr,
    dA_cumsum_ptr, cb_ptr, ddAcs_ptr, chunk_size, hdim, batch, seqlen,
    nheads_ngroups_ratio, stride_x_batch, stride_x_seqlen, stride_x_head,
    stride_x_hdim, stride_dout_batch, stride_dout_seqlen, stride_dout_head,
    stride_dout_hdim, stride_dt_batch, stride_dt_chunk, stride_dt_head,
    stride_dt_csize, stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dA_cs_head, stride_dA_cs_csize, stride_cb_batch, stride_cb_chunk,
    stride_cb_head, stride_cb_csize_m, stride_cb_csize_n,
    stride_ddAcs_batch, stride_ddAcs_chunk, stride_ddAcs_head,
    stride_ddAcs_csize_m, stride_ddAcs_csize_n, BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'
    ):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += (pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen +
        pid_h * stride_x_head)
    dout_ptr += (pid_b * stride_dout_batch + pid_c * chunk_size *
        stride_dout_seqlen + pid_h * stride_dout_head)
    dt_ptr += (pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h *
        stride_dt_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + pid_h * stride_dA_cs_head)
    cb_ptr += (pid_b * stride_cb_batch + pid_c * stride_cb_chunk + pid_h //
        nheads_ngroups_ratio * stride_cb_head)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[
        None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] *
        stride_x_hdim)
    dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None,
        :] * stride_cb_csize_n)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    chunk_size_limit_n = min(chunk_size_limit, (pid_m + 1) * BLOCK_SIZE_M)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_k[None, :] < hdim), other=0.0)
    x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] <
        chunk_size_limit_n), other=0.0)
    acc = tl.dot(dout, x)
    cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[
        None, :] < chunk_size), other=0.0)
    acc *= cb
    dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size, other=0.0)
    acc *= dt_n
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=
        offs_m < chunk_size, other=0.0)
    dA_cs_n = tl.load(dA_cumsum_ptr + offs_n * stride_dA_cs_csize, mask=
        offs_n < chunk_size, other=0.0)
    acc *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
    mask = offs_m[:, None] >= offs_n[None, :] + 1
    acc = tl.where(mask, acc, 0.0)
    acc = tl.cumsum(acc, axis=1)
    acc = tl.where(mask, acc, 0.0)
    ddA_cs = tl.sum(acc, axis=0)
    ddAcs_ptr += (pid_b * stride_ddAcs_batch + pid_c * stride_ddAcs_chunk +
        pid_h * stride_ddAcs_head + pid_m * stride_ddAcs_csize_m)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ddAcs_ptrs = ddAcs_ptr + offs_n * stride_ddAcs_csize_n
    tl.store(ddAcs_ptrs + stride_ddAcs_csize_n, ddA_cs, mask=offs_n < 
        chunk_size - 1)
    tl.store(ddAcs_ptr, 0.0)
