import triton
import triton.language as tl
import torch

@triton.jit
def _dq_prob_bwd_kernel(Q, K, dQ, LSE, dLSE, nheads, seqlen_q, seqlen_k,
    BLOCK_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr'):
    ASM: 'tl.constexpr' = 'cvt.rna.tf32.f32 $0, $1;'
    start_m = tl.program_id(0)
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + ndims * offs_m[:, None]
    dq_ptrs = dQ + ndims * offs_m[:, None]
    k_ptrs = K + ndims * offs_n[:, None]
    lse = tl.load(LSE + offs_m, mask=offs_m < seqlen_q, other=0.0)
    dlse = tl.load(dLSE + offs_m, mask=offs_m < seqlen_q, other=0.0)
    end_n = seqlen_k
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q,
                other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n +
                offs_n)[:, None] < seqlen_k, other=0.0)
            qk += tl.dot(q, tl.trans(k))
        qk_grad = tl.exp(qk - lse[:, None])
        qk_grad = tl.where((start_n + offs_n)[None, :] < seqlen_k, qk_grad, 0.0
            )
        qk_grad = qk_grad * dlse[:, None]
        qk_grad = tl.inline_asm_elementwise(ASM, '=r, r', [qk_grad], dtype=
            tl.float32, is_pure=True, pack=1)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            q = tl.load(q_ptrs + offs_hd, mask=offs_m[:, None] < seqlen_q,
                other=0.0)
            k = tl.load(k_ptrs + offs_hd + start_n * ndims, mask=(start_n +
                offs_n)[:, None] < seqlen_k, other=0.0)
            k = tl.inline_asm_elementwise(ASM, '=r, r', [k], dtype=tl.
                float32, is_pure=True, pack=1)
            q_grad = tl.dot(qk_grad, k)
            dq_h = tl.load(dq_ptrs + offs_hd, mask=offs_m[:, None] <
                seqlen_q, other=0.0)
            tl.store(dq_ptrs + offs_hd, dq_h + q_grad, mask=offs_m[:, None] <
                seqlen_q)
