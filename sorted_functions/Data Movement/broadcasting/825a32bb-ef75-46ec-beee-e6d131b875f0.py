import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_HEADDIM': lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM'], 'EVEN_V_HEADDIM': lambda args: args['v_headdim'] ==
    args['V_BLOCK_HEADDIM']})
@triton.jit
def _bwd_permuted_block_diagonal_kernel(Q, K, V, q_sort_idx, k_sort_idx, DO,
    DQ, DK, DV, LSE, D, softmax_scale, stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn,
    stride_q_sort_idxb, stride_q_sort_idxh, stride_q_sort_idxm,
    stride_k_sort_idxb, stride_k_sort_idxh, stride_k_sort_idxn, stride_dob,
    stride_doh, stride_dom, stride_dqb, stride_dqh, stride_dqm, stride_dkb,
    stride_dkh, stride_dkn, stride_dvb, stride_dvh, stride_dvn, nheads,
    seqlen_q, block_size, headdim, v_headdim, smooth_block,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BLOCK_HEADDIM: 'tl.constexpr',
    V_BLOCK_HEADDIM: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr',
    EVEN_V_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr'):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    Q_idx = (q_sort_idx + off_b * stride_q_sort_idxb + off_h *
        stride_q_sort_idxh)
    K_idx = (k_sort_idx + off_b * stride_k_sort_idxb + off_h *
        stride_k_sort_idxh)
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    D += off_hb * seqlen_q
    LSE += off_hb * seqlen_q
    start_n = tl.program_id(0)
    _bwd_blocked_kernel_one_col(start_n=start_n, Q=Q, K=K, V=V, Q_idx=Q_idx,
        K_idx=K_idx, DO=DO, DQ=DQ, DK=DK, DV=DV, LSE=LSE, D=D,
        softmax_scale=softmax_scale, stride_qm=stride_qm, stride_kn=
        stride_kn, stride_vn=stride_vn, stride_dom=stride_dom, stride_dqm=
        stride_dqm, stride_dkn=stride_dkn, stride_dvn=stride_dvn,
        stride_q_idxm=stride_q_sort_idxm, stride_k_idxn=stride_k_sort_idxn,
        seqlen_q=seqlen_q, block_size=block_size // BLOCK_N, headdim=
        headdim, v_headdim=v_headdim, smooth_block=smooth_block,
        BLOCK_HEADDIM=BLOCK_HEADDIM, V_BLOCK_HEADDIM=V_BLOCK_HEADDIM,
        EVEN_HEADDIM=EVEN_HEADDIM, EVEN_V_HEADDIM=EVEN_V_HEADDIM, BLOCK_M=
        BLOCK_M, BLOCK_N=BLOCK_N)
