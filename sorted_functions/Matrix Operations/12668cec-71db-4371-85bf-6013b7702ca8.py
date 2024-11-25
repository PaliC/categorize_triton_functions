import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BT', 'BK'])
@triton.jit
def fwd_recompute_w_kernel(k, beta, w, A, s_qk_h, s_qk_t, s_qk_d, T, K, BT:
    'tl.constexpr', BK: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (
        BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT,
        0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_w = tl.dot(b_A, b_kb, allow_tf32=False)
        p_w = tl.make_block_ptr(w + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_w, b_w, boundary_check=(0, 1))
