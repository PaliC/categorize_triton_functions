import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,
    'SEQUENCE_PARALLEL': False}, num_warps=8, num_stages=1, pre_hook=
    init_to_zero('DQ')), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,
    'SEQUENCE_PARALLEL': True}, num_warps=8, num_stages=1, pre_hook=
    init_to_zero('DQ'))], key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K',
    'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM'])
@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args[
    'BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args[
    'BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM']})
@triton.jit
def _bwd_kernel(Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm,
    stride_dob, stride_doh, stride_dom, stride_dqb, stride_dqh, stride_dqm,
    stride_dkb, stride_dkh, stride_dkn, stride_dvb, stride_dvh, stride_dvn,
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: 'tl.constexpr',
    IS_CAUSAL: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr',
    SEQUENCE_PARALLEL: 'tl.constexpr', EVEN_M: 'tl.constexpr', EVEN_N:
    'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr'):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != 'none':
        Bias += off_b * stride_bb + off_h * stride_bh
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK,
                DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn,
                stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn,
                seqlen_q, seqlen_k, headdim, ATOMIC_ADD=False, BIAS_TYPE=
                BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV,
            LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn,
            stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn,
            seqlen_q, seqlen_k, headdim, ATOMIC_ADD=True, BIAS_TYPE=
            BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
