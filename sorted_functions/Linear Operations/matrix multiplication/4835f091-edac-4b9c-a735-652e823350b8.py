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
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 
    32}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_M': 64,
    'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2)],
    key=['chunk_size', 'K', 'IS_CAUSAL'])
@triton.jit
def _bmm_chunk_fwd_kernel(a_ptr, b_ptr, out_ptr, seq_idx_ptr, seqlen,
    chunk_size, K, ngroups, stride_a_batch, stride_a_seqlen, stride_a_head,
    stride_ak, stride_b_batch, stride_b_seqlen, stride_b_head, stride_bk,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_outm,
    stride_outn, stride_seq_idx_batch, stride_seq_idx_seqlen, IS_CAUSAL:
    'tl.constexpr', dot_dtype: 'tl.constexpr', HAS_SEQ_IDX: 'tl.constexpr',
    BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr',
    BLOCK_SIZE_K: 'tl.constexpr'):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += (pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen +
        pid_h * stride_a_head)
    b_ptr += (pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen +
        pid_h * stride_b_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch + pid_c * chunk_size *
            stride_seq_idx_seqlen)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] *
        stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] *
        stride_b_seqlen)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (
            offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) &
            (offs_n[None, :] < chunk_size_limit), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen,
            mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc
    out_ptr += (pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h *
        stride_out_head)
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] *
        stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[
        None, :] < chunk_size))
