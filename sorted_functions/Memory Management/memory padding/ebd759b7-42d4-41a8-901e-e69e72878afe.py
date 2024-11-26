import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K':
    64}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 
    32}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2)],
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'])
@triton.jit
def _chunk_scan_fwd_kernel(cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr, dt_ptr,
    dA_cumsum_ptr, seq_idx_ptr, C_ptr, prev_states_ptr, D_ptr, chunk_size,
    hdim, dstate, batch, seqlen, nheads_ngroups_ratio, stride_cb_batch,
    stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_dA_cs_csize, stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head,
    stride_states_hdim, stride_states_dstate, stride_D_head, IS_CAUSAL:
    'tl.constexpr', HAS_D: 'tl.constexpr', D_HAS_HDIM: 'tl.constexpr',
    HAS_Z: 'tl.constexpr', HAS_SEQ_IDX: 'tl.constexpr', BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K:
    'tl.constexpr', BLOCK_SIZE_DSTATE: 'tl.constexpr', IS_TRITON_22:
    'tl.constexpr'):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
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
    prev_states_ptr += (pid_b * stride_states_batch + pid_c *
        stride_states_chunk + pid_h * stride_states_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=
        offs_m < chunk_size, other=0.0)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=
            pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit, other=-1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if IS_TRITON_22 or pid_c > -1:
        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <=
            128 else BLOCK_SIZE_K)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate
            [None, :] * stride_C_dstate)
        prev_states_ptrs = prev_states_ptr + (offs_n[None, :] *
            stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
                (offs_k_dstate[None, :] < dstate), other=0.0)
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:,
                None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            prev_states = prev_states
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(C_ptrs, mask=(offs_m[:, None] <
                    chunk_size_limit) & (offs_k_dstate[None, :] < dstate -
                    k), other=0.0)
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate
                    [:, None] < dstate - k) & (offs_n[None, :] < hdim),
                    other=0.0)
                prev_states = prev_states
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m[:, None]
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None,
        :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] *
        stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) *
        BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k
            [None, :] < chunk_size - k), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k,
            other=0.0)
        cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) &
            (offs_n[None, :] < hdim), other=0.0)
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n <
                hdim, other=0.0)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head)
        x_residual = tl.load(x_ptr + (offs_m[:, None] * stride_x_seqlen + 
            offs_n[None, :] * stride_x_hdim), mask=(offs_m[:, None] <
            chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0)
        acc += x_residual * D
    if HAS_Z:
        out_x_ptr += (pid_b * stride_out_batch + pid_c * chunk_size *
            stride_out_seqlen + pid_h * stride_out_head)
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] +
            offs_out_n[None, :])
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] <
            chunk_size_limit) & (offs_out_n[None, :] < hdim))
        z_ptr += (pid_b * stride_z_batch + pid_c * chunk_size *
            stride_z_seqlen + pid_h * stride_z_head)
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + 
            stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) &
            (offs_out_n[None, :] < hdim), other=0.0)
        acc *= z * tl.sigmoid(z)
    out_ptr += (pid_b * stride_out_batch + pid_c * chunk_size *
        stride_out_seqlen + pid_h * stride_out_head)
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + 
        offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) &
        (offs_out_n[None, :] < hdim))