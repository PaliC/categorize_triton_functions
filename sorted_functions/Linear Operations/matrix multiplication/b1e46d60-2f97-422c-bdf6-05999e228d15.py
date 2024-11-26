import triton
import triton.language as tl
import torch

@triton_autotune(configs=_get_bmm_configs(), key=['M', 'N',
    'AUTOTUNE_MAX_SEQ_LEN'])
@triton.jit
def _jagged_jagged_bmm_reduce_sum(seq_offsets, JaggedA, JaggedB, Out,
    ReduceOut, M, N, AUTOTUNE_MAX_SEQ_LEN, stride_ak, stride_bk, stride_ob,
    stride_om, stride_on, stride_orb, stride_orn, REDUCE_JAGGEDB:
    'tl.constexpr', ALLOW_TF32: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr'):
    """
    Computing bmm Out = Jagged x Jagged
    K is the jagged dimension
    JaggedA has shape (sum_B(K_i), M), JaggedB has shape (sum_B(K_i), N), and Out has shape (B, M, N)
    """
    off_b = tl.program_id(0)
    off_m = tl.program_id(1)
    off_n = tl.program_id(2)
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Out += off_b * stride_ob
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if REDUCE_JAGGEDB:
        out_reduce_ptrs = ReduceOut + off_b * stride_orb + offs_n * stride_orn
        acc_reduce = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if seq_len == 0:
        out = accumulator
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None,
            :] < N))
        if REDUCE_JAGGEDB:
            if off_m == 0:
                tl.store(out_reduce_ptrs, acc_reduce, mask=offs_n < N)
        return
    JaggedA += seq_start * stride_ak
    JaggedB += seq_start * stride_bk
    offs_k = tl.arange(0, BLOCK_K)
    jg_a_ptrs = JaggedA + offs_k[None, :] * stride_ak + offs_m[:, None]
    jg_b_ptrs = JaggedB + offs_k[:, None] * stride_bk + offs_n[None, :]
    for k in range(0, seq_len, BLOCK_K):
        jg_a = tl.load(jg_a_ptrs, mask=offs_m[:, None] < M and (k + offs_k)
            [None, :] < seq_len, other=0.0)
        jg_b = tl.load(jg_b_ptrs, mask=offs_n[None, :] < N and (k + offs_k)
            [:, None] < seq_len, other=0.0)
        accumulator += tl.dot(jg_a, jg_b, allow_tf32=ALLOW_TF32)
        if REDUCE_JAGGEDB:
            if off_m == 0:
                acc_reduce += tl.sum(jg_b, axis=0)
        jg_a_ptrs += BLOCK_K * stride_ak
        jg_b_ptrs += BLOCK_K * stride_bk
    out = accumulator
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    if REDUCE_JAGGEDB:
        if off_m == 0:
            tl.store(out_reduce_ptrs, acc_reduce, mask=offs_n < N)
