import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_N': 64}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_N': 32}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_N': 64}, num_stages=4,
    num_warps=8), triton.Config({'BLOCK_SIZE_N': 32}, num_stages=4,
    num_warps=8)], key=['chunk_size', 'hdim', 'dstate'])
@triton.jit
def _chunk_scan_fwd_kernel_wip(cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr,
    dt_ptr, dA_cumsum_ptr, seq_idx_ptr, C_ptr, B_ptr, prev_states_ptr,
    D_ptr, chunk_size, hdim, dstate, batch, seqlen, nheads_ngroups_ratio,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m,
    stride_cb_csize_k, stride_x_batch, stride_x_seqlen, stride_x_head,
    stride_x_hdim, stride_z_batch, stride_z_seqlen, stride_z_head,
    stride_z_hdim, stride_out_batch, stride_out_seqlen, stride_out_head,
    stride_out_hdim, stride_dt_batch, stride_dt_chunk, stride_dt_head,
    stride_dt_csize, stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dA_cs_head, stride_dA_cs_csize, stride_seq_idx_batch,
    stride_seq_idx_seqlen, stride_C_batch, stride_C_seqlen, stride_C_head,
    stride_C_dstate, stride_B_batch, stride_B_seqlen, stride_B_head,
    stride_B_dstate, stride_states_batch, stride_states_chunk,
    stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head, HAS_D: 'tl.constexpr', D_HAS_HDIM: 'tl.constexpr', HAS_Z:
    'tl.constexpr', HAS_SEQ_IDX: 'tl.constexpr', BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_DSTATE:
    'tl.constexpr'):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_n = tl.program_id(axis=0)
    cb_ptr += (pid_b * stride_cb_batch + pid_c * stride_cb_chunk + pid_h //
        nheads_ngroups_ratio * stride_cb_head)
    x_ptr += (pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen +
        pid_h * stride_x_head)
    dt_ptr += (pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h *
        stride_dt_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + pid_h * stride_dA_cs_head)
    C_ptr += (pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen +
        pid_h // nheads_ngroups_ratio * stride_C_head)
    B_ptr += (pid_b * stride_B_batch + pid_c * chunk_size * stride_B_seqlen +
        pid_h // nheads_ngroups_ratio * stride_B_head)
    prev_states_ptr += (pid_b * stride_states_batch + pid_c *
        stride_states_chunk + pid_h * stride_states_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    out_ptr += (pid_b * stride_out_batch + pid_c * chunk_size *
        stride_out_seqlen + pid_h * stride_out_head)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[
        None, :] * stride_C_dstate)
    B_ptrs = B_ptr + (offs_m[None, :] * stride_B_seqlen + offs_k_dstate[:,
        None] * stride_B_dstate)
    prev_states_ptrs = prev_states_ptr + (offs_n[None, :] *
        stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_m[None,
        :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] *
        stride_x_hdim)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None,
        :] * stride_out_hdim)
    prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] <
        dstate) & (offs_n[None, :] < hdim), other=0.0)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    for start_m in range(0, chunk_size_limit, BLOCK_SIZE_M):
        start_m = tl.multiple_of(start_m, BLOCK_SIZE_M)
        dA_cs_m = tl.load(dA_cumsum_ptr + (start_m + offs_m) *
            stride_dA_cs_csize, mask=offs_m < chunk_size - start_m, other=0.0)
        if HAS_SEQ_IDX:
            seq_idx_prev = tl.load(seq_idx_ptr + start_m -
                stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
            seq_idx_m = tl.load(seq_idx_ptr + (start_m + offs_m) *
                stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit -
                start_m, other=-1)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit -
            start_m) & (offs_k_dstate[None, :] < dstate), other=0.0)
        acc = tl.dot(C, prev_states) * scale_m[:, None]
        dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size - start_m, other=0.0)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit -
            start_m) & (offs_n[None, :] < hdim), other=0.0)
        if HAS_D:
            if D_HAS_HDIM:
                D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=
                    offs_n < hdim, other=0.0)
            else:
                D = tl.load(D_ptr + pid_h * stride_D_head)
            acc += x * D
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit -
            start_m) & (offs_n[None, :] < hdim))
        if start_m + BLOCK_SIZE_M < chunk_size_limit:
            B = tl.load(B_ptrs, mask=(offs_m[None, :] < chunk_size_limit -
                start_m) & (offs_k_dstate[:, None] < dstate), other=0.0)
            dA_cs_last = tl.load(dA_cumsum_ptr + (start_m + BLOCK_SIZE_M) *
                stride_dA_cs_csize)
            scale = tl.exp(dA_cs_last - dA_cs_m) * dt_m
            B = B
            tmp = tl.dot(B, x)
            prev_states += tmp
        C_ptrs += BLOCK_SIZE_M * stride_C_seqlen
        B_ptrs += BLOCK_SIZE_M * stride_B_seqlen
        cb_ptrs += (BLOCK_SIZE_M * stride_cb_csize_m + BLOCK_SIZE_M *
            stride_cb_csize_k)
        x_ptrs += BLOCK_SIZE_M * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_M * stride_dt_csize
        out_ptrs += BLOCK_SIZE_M * stride_out_seqlen
