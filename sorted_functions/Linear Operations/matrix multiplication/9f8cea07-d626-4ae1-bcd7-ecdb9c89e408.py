import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K':
    32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(['ddA_cumsum_ptr'
    ])), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages
    =3, num_warps=4, pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.
    Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3,
    num_warps=4, pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config(
    {'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4,
    pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8,
    pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8,
    pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8,
    pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8,
    pre_hook=init_to_zero(['ddA_cumsum_ptr']))], key=['chunk_size', 'hdim',
    'dstate'])
@triton.jit
def _chunk_state_bwd_ddAcs_stable_kernel(x_ptr, b_ptr, dstates_ptr, dt_ptr,
    dA_cumsum_ptr, seq_idx_ptr, ddA_cumsum_ptr, chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio, stride_x_batch, stride_x_seqlen,
    stride_x_head, stride_x_hdim, stride_b_batch, stride_b_seqlen,
    stride_b_head, stride_b_dstate, stride_dstates_batch,
    stride_dstates_chunk, stride_states_head, stride_states_hdim,
    stride_states_dstate, stride_dt_batch, stride_dt_chunk, stride_dt_head,
    stride_dt_csize, stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dA_cs_head, stride_dA_cs_csize, stride_seq_idx_batch,
    stride_seq_idx_seqlen, stride_ddA_cs_batch, stride_ddA_cs_chunk,
    stride_ddA_cs_head, stride_ddA_cs_csize, HAS_SEQ_IDX: 'tl.constexpr',
    BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr',
    BLOCK_SIZE_K: 'tl.constexpr', BLOCK_SIZE_DSTATE: 'tl.constexpr'):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += (pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen +
        pid_h * stride_x_head)
    b_ptr += (pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen +
        pid_h // nheads_ngroups_ratio * stride_b_head)
    dstates_ptr += (pid_b * stride_dstates_batch + pid_c *
        stride_dstates_chunk + pid_h * stride_states_head)
    dt_ptr += (pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h *
        stride_dt_head)
    ddA_cumsum_ptr += (pid_b * stride_ddA_cs_batch + pid_c *
        stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + pid_h * stride_dA_cs_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    offs_k = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else
        BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_k[None, :] *
        stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_hdim + 
        offs_k[:, None] * stride_states_dstate)
    if BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_k[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate) & (
            offs_n[None, :] < hdim), other=0.0)
        dstates = dstates
        acc = tl.dot(b, dstates)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
                (offs_k[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate -
                k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_states_dstate
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=
        offs_m < chunk_size, other=0.0)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize)
    if not HAS_SEQ_IDX:
        scale = tl.exp(dA_cs_last - dA_cs_m)
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) *
            stride_seq_idx_seqlen)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last -
            dA_cs_m), 0.0)
    acc *= scale[:, None]
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] *
        stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n
        [None, :] < hdim), other=0.0)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0)
    ddt = tl.sum(acc * x, axis=1)
    ddA_cs = ddt * dt_m
    ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    tl.atomic_add(ddA_cumsum_ptrs + stride_ddA_cs_csize, ddA_cs, mask=
        offs_m < chunk_size - 1)
