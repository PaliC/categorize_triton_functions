import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 64}), triton.Config(
    {'BLOCK_SIZE': 128}), triton.Config({'BLOCK_SIZE': 256}), triton.Config
    ({'BLOCK_SIZE': 512}), triton.Config({'BLOCK_SIZE': 1024}), triton.
    Config({'BLOCK_SIZE': 2048})], key=['dim'])
@triton.jit
def _state_passing_fwd_kernel(states_ptr, out_ptr, final_states_ptr,
    dA_cs_ptr, initstates_ptr, seq_idx_ptr, dim, nchunks, seqlen,
    chunk_size, stride_states_batch, stride_states_chunk,
    stride_states_head, stride_states_dim, stride_out_batch,
    stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head,
    stride_final_states_dim, stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dA_cs_head, stride_initstates_batch, stride_initstates_head,
    stride_initstates_dim, stride_seq_idx_batch, stride_seq_idx_seqlen,
    HAS_INITSTATES: 'tl.constexpr', HAS_SEQ_IDX: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr'):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += (pid_b * stride_final_states_batch + pid_h *
        stride_final_states_head)
    if HAS_INITSTATES:
        initstates_ptr += (pid_b * stride_initstates_batch + pid_h *
            stride_initstates_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim
    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    else:
        initstates_ptrs = initstates_ptr + offs_m * stride_initstates_dim
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0)
    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    seq_idx = 0
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0)
        dA_cs = tl.load(dA_cs_ptr)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size,
                seqlen) - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk