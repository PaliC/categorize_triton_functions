import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 64}), triton.Config(
    {'BLOCK_SIZE': 128}), triton.Config({'BLOCK_SIZE': 256}), triton.Config
    ({'BLOCK_SIZE': 512}), triton.Config({'BLOCK_SIZE': 1024}), triton.
    Config({'BLOCK_SIZE': 2048})], key=['dim'])
@triton.jit
def _state_passing_bwd_kernel(dout_ptr, out_ptr, dA_cs_ptr,
    dfinal_states_ptr, seq_idx_ptr, dstates_ptr, ddA_cs_ptr,
    dinitstates_ptr, states_converted_ptr, dim, nchunks, seqlen, chunk_size,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_dfinal_states_batch, stride_dfinal_states_head,
    stride_dfinal_states_dim, stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head,
    stride_dstates_dim, stride_ddA_cs_batch, stride_ddA_cs_chunk,
    stride_ddA_cs_head, stride_dinitstates_batch, stride_dinitstates_head,
    stride_dinitstates_dim, CONVERT_STATES: 'tl.constexpr',
    HAS_DFINAL_STATES: 'tl.constexpr', HAS_DINITSTATES: 'tl.constexpr',
    HAS_SEQ_IDX: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    dstates_ptr += (pid_b * stride_dstates_batch + pid_h *
        stride_dstates_head + (nchunks - 1) * stride_dstates_chunk)
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head + (
        nchunks - 1) * stride_dA_cs_chunk
    ddA_cs_ptr += pid_b * stride_ddA_cs_batch + pid_h * stride_ddA_cs_head + (
        nchunks - 1) * stride_ddA_cs_chunk + pid_m
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head + (nchunks -
        1) * stride_out_chunk
    dout_ptr += pid_b * stride_dout_batch + pid_h * stride_dout_head + (nchunks
         - 1) * stride_dout_chunk
    if CONVERT_STATES:
        states_converted_ptr += (pid_b * stride_out_batch + pid_h *
            stride_out_head + (nchunks - 1) * stride_out_chunk)
    if HAS_DFINAL_STATES:
        dfinal_states_ptr += (pid_b * stride_dfinal_states_batch + pid_h *
            stride_dfinal_states_head)
    if HAS_DINITSTATES:
        dinitstates_ptr += (pid_b * stride_dinitstates_batch + pid_h *
            stride_dinitstates_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dstates_ptrs = dstates_ptr + offs_m * stride_dstates_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    dout_ptrs = dout_ptr + offs_m * stride_dout_dim
    if CONVERT_STATES:
        states_converted_ptrs = states_converted_ptr + offs_m * stride_out_dim
    if HAS_DFINAL_STATES:
        dstates = tl.load(dfinal_states_ptr + offs_m *
            stride_dfinal_states_dim, mask=offs_m < dim, other=0.0)
    else:
        dstates = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + (seqlen - 1) * stride_seq_idx_seqlen)
    dstates_ptrs -= stride_dstates_chunk
    for c in range(nchunks - 1):
        dA_cs = tl.load(dA_cs_ptr)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + ((nchunks - c - 1) *
                chunk_size - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0)
        if CONVERT_STATES:
            tl.store(states_converted_ptrs, out, mask=offs_m < dim)
        ddA = tl.sum(out * dstates) * scale
        tl.store(ddA_cs_ptr, ddA)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0)
        dstates = scale * dstates + dout
        tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
        dout_ptrs -= stride_dout_chunk
        dstates_ptrs -= stride_dstates_chunk
        dA_cs_ptr -= stride_dA_cs_chunk
        ddA_cs_ptr -= stride_ddA_cs_chunk
        out_ptrs -= stride_out_chunk
        if CONVERT_STATES:
            states_converted_ptrs -= stride_out_chunk
    if CONVERT_STATES:
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0)
        tl.store(states_converted_ptrs, out, mask=offs_m < dim)
    if not HAS_DINITSTATES:
        tl.store(ddA_cs_ptr, 0.0)
    else:
        dA_cs = tl.load(dA_cs_ptr)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            scale = tl.where(seq_idx == 0, scale, 0.0)
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0)
        ddA = tl.sum(out * dstates) * scale
        tl.store(ddA_cs_ptr, ddA)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0)
        dstates = scale * dstates + dout
        tl.store(dinitstates_ptr + offs_m * stride_dinitstates_dim, dstates,
            mask=offs_m < dim)
