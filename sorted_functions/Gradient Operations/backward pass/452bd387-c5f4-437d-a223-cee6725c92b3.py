import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N':
    128}, num_stages=3, num_warps=4, pre_hook=init_to_zero([
    'ddA_cumsum_ptr'])), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':
    32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(['ddA_cumsum_ptr'
    ])), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128},
    num_stages=3, num_warps=4, pre_hook=init_to_zero(['ddA_cumsum_ptr'])),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3,
    num_warps=4, pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config(
    {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4,
    pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4,
    pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4,
    pre_hook=init_to_zero(['ddA_cumsum_ptr'])), triton.Config({
    'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4,
    pre_hook=init_to_zero(['ddA_cumsum_ptr']))], key=['chunk_size',
    'dstate', 'hdim'])
@triton.jit
def _chunk_scan_bwd_dc_kernel(dout_ptr, prev_states_ptr, C_ptr,
    dA_cumsum_ptr, seq_idx_ptr, dc_ptr, ddA_cumsum_ptr, chunk_size, dstate,
    hdim, batch, seqlen, nheads, nheads_per_program, ngroups,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head,
    stride_dout_hdim, stride_prev_states_batch, stride_prev_states_chunk,
    stride_prev_states_head, stride_prev_states_hdim,
    stride_prev_states_dstate, stride_C_batch, stride_C_seqlen,
    stride_C_head, stride_C_dstate, stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dA_cs_head, stride_dA_cs_csize, stride_seq_idx_batch,
    stride_seq_idx_seqlen, stride_dc_batch, stride_dc_seqlen,
    stride_dc_split, stride_dc_group, stride_dc_dstate, stride_ddA_cs_batch,
    stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    HAS_DDA_CS: 'tl.constexpr', HAS_SEQ_IDX: 'tl.constexpr', BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'
    ):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(axis=2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    dout_ptr += (pid_b * stride_dout_batch + pid_c * chunk_size *
        stride_dout_seqlen + (pid_g * (nheads // ngroups) + pid_s *
        nheads_per_program) * stride_dout_head)
    dc_ptr += (pid_b * stride_dc_batch + pid_c * chunk_size *
        stride_dc_seqlen + pid_g * stride_dc_group + pid_s * stride_dc_split)
    prev_states_ptr += (pid_b * stride_prev_states_batch + pid_c *
        stride_prev_states_chunk + (pid_g * (nheads // ngroups) + pid_s *
        nheads_per_program) * stride_prev_states_head)
    dA_cumsum_ptr += (pid_b * stride_dA_cs_batch + pid_c *
        stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s *
        nheads_per_program) * stride_dA_cs_head)
    if HAS_DDA_CS:
        C_ptr += (pid_b * stride_C_batch + pid_c * chunk_size *
            stride_C_seqlen + pid_g * stride_C_head)
        ddA_cumsum_ptr += (pid_b * stride_ddA_cs_batch + pid_c *
            stride_ddA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s *
            nheads_per_program) * stride_ddA_cs_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[
        None, :] * stride_dout_hdim)
    prev_states_ptrs = prev_states_ptr + (offs_n[None, :] *
        stride_prev_states_dstate + offs_k[:, None] * stride_prev_states_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    if HAS_DDA_CS:
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_n[None,
            :] * stride_C_dstate)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        c = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_n[None, :] < dstate), other=0.0)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=
            pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit, other=-1)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s *
        nheads_per_program)
    for h in range(nheads_iter):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
            (offs_k[None, :] < hdim), other=0.0)
        prev_states = tl.load(prev_states_ptrs, mask=(offs_k[:, None] <
            hdim) & (offs_n[None, :] < dstate), other=0.0)
        prev_states = prev_states
        dc = tl.dot(dout, prev_states)
        dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size_limit,
            other=0.0)
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_m)
        else:
            scale = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        dc *= scale[:, None]
        if HAS_DDA_CS:
            ddA_cs = tl.sum(dc * c, axis=1)
            tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
        acc += dc
        dout_ptrs += stride_dout_head
        prev_states_ptrs += stride_prev_states_head
        dA_cumsum_ptrs += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptrs += stride_ddA_cs_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dc_ptrs = dc_ptr + (offs_m[:, None] * stride_dc_seqlen + offs_n[None, :
        ] * stride_dc_dstate)
    tl.store(dc_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (
        offs_n[None, :] < dstate))
