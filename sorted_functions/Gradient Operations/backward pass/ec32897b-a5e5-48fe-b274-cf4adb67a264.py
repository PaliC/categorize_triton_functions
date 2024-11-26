import triton
import triton.language as tl
import torch

@triton.jit
def _dk_prob_bwd_kernel(Q, K, dK, LSE, dLSE, nheads, seqlen_q, seqlen_k,
    BLOCK_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr'):
    ASM: 'tl.constexpr' = 'cvt.rna.tf32.f32 $0, $1;'
    start_n = tl.program_id(0)
    ndims = nheads * BLOCK_HEADDIM
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) + start_n * BLOCK_N
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + ndims * offs_m[:, None]
    k_ptrs = K + ndims * offs_n[:, None]
    dk_ptrs = dK + ndims * offs_n[:, None]
    end_m = seqlen_q
    for start_m in range(0, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        lse = tl.load(LSE + offs_m + start_m, mask=offs_m < seqlen_q, other=0.0
            )
        dlse = tl.load(dLSE + offs_m + start_m, mask=offs_m < seqlen_q,
            other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            q = tl.load(q_ptrs + offs_hd + start_m * ndims, mask=(offs_m +
                start_m)[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd, mask=offs_n[:, None] < seqlen_k,
                other=0.0)
            qk += tl.dot(q, tl.trans(k))
        qk_grad = tl.exp(qk - lse[:, None])
        qk_grad = tl.where((start_m + offs_m)[:, None] < seqlen_q, qk_grad, 0.0
            )
        qk_grad = qk_grad * dlse[:, None]
        qk_grad = tl.inline_asm_elementwise(ASM, '=r, r', [qk_grad], dtype=
            tl.float32, is_pure=True, pack=1)
        for off_h in range(nheads):
            offs_hd = (offs_d + off_h * BLOCK_HEADDIM)[None, :]
            q = tl.load(q_ptrs + offs_hd + start_m * ndims, mask=(start_m +
                offs_m)[:, None] < seqlen_q, other=0.0)
            k = tl.load(k_ptrs + offs_hd, mask=offs_n[:, None] < seqlen_k,
                other=0.0)
            q = tl.inline_asm_elementwise(ASM, '=r, r', [q], dtype=tl.
                float32, is_pure=True, pack=1)
            k_grad = tl.dot(tl.trans(qk_grad), q)
            dk_h = tl.load(dk_ptrs + offs_hd, mask=offs_n[:, None] <
                seqlen_k, other=0.0)
            tl.store(dk_ptrs + offs_hd, dk_h + k_grad, mask=offs_n[:, None] <
                seqlen_k)
