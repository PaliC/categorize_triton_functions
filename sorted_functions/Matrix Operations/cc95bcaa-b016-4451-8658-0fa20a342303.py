import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BM': 128, 'BK': 64, 'BN': 256,
    'G': 4}, num_stages=3, num_warps=8), triton.Config({'BM': 64, 'BK': 32,
    'BN': 256, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 
    128, 'BK': 32, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4), triton.
    Config({'BM': 128, 'BK': 32, 'BN': 64, 'G': 4}, num_stages=4, num_warps
    =4), triton.Config({'BM': 64, 'BK': 32, 'BN': 128, 'G': 4}, num_stages=
    4, num_warps=4), triton.Config({'BM': 128, 'BK': 32, 'BN': 32, 'G': 4},
    num_stages=4, num_warps=4), triton.Config({'BM': 64, 'BK': 32, 'BN': 32,
    'G': 4}, num_stages=5, num_warps=2), triton.Config({'BM': 32, 'BK': 32,
    'BN': 64, 'G': 4}, num_stages=5, num_warps=2), triton.Config({'BM': 128,
    'BK': 128, 'BN': 256, 'G': 4}, num_stages=3, num_warps=8), triton.
    Config({'BM': 256, 'BK': 128, 'BN': 128, 'G': 4}, num_stages=3,
    num_warps=8), triton.Config({'BM': 256, 'BK': 128, 'BN': 64, 'G': 4},
    num_stages=4, num_warps=4), triton.Config({'BM': 64, 'BK': 128, 'BN': 
    256, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 128,
    'BK': 128, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4), triton.
    Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps
    =4), triton.Config({'BM': 64, 'BK': 64, 'BN': 128, 'G': 4}, num_stages=
    4, num_warps=4), triton.Config({'BM': 128, 'BK': 64, 'BN': 32, 'G': 4},
    num_stages=4, num_warps=4)], key=['M', 'N', 'K'])
@triton.heuristics({'HAS_INPUT': lambda args: args['input'] is not None,
    'HAS_ALPHA': lambda args: args['alpha'] is not None, 'HAS_BETA': lambda
    args: args['beta'] is not None})
@triton.jit
def matmul_kernel(a, b, c, input, alpha, beta, M, N, K, s_am, s_ak, s_bk,
    s_bn, s_cm, s_cn, BM: 'tl.constexpr', BK: 'tl.constexpr', BN:
    'tl.constexpr', G: 'tl.constexpr', ACTIVATION: 'tl.constexpr',
    HAS_INPUT: 'tl.constexpr', HAS_ALPHA: 'tl.constexpr', HAS_BETA:
    'tl.constexpr'):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    NM, NN = tl.num_programs(0), tl.num_programs(1)
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    i_m, i_n = tl.swizzle2d(i_m, i_n, NM, NN, G)
    o_am = (i_m * BM + tl.arange(0, BM)) % M
    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)
    p_a = a + (o_am[:, None] * s_am + o_k[None, :] * s_ak)
    p_b = b + (o_k[:, None] * s_bk + o_bn[None, :] * s_bn)
    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)
        b_acc += tl.dot(b_a, b_b, allow_tf32=False)
        p_a += BK * s_ak
        p_b += BK * s_bk
    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    mask = (o_cm[:, None] < M) & (o_cn[None, :] < N)
    b_c = b_acc
    if ACTIVATION == 'leaky_relu':
        b_c = leaky_relu(b_c)
    if HAS_ALPHA:
        b_c *= tl.load(alpha)
    if HAS_INPUT:
        p_i = input + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]
        b_i = tl.load(p_i, mask=mask, other=0.0)
        if HAS_BETA:
            b_i *= tl.load(beta)
        b_c += b_i
    p_c = c + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]
    tl.store(p_c, b_c, mask=mask)
