import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16)], key=['BK'])
@triton.jit
def fwd_prepare_wy_repr_kernel_chunk32(k, beta, A, T: 'tl.constexpr', H:
    'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BK:
    'tl.constexpr', BC: 'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if HEAD_FIRST:
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,),
            (BT,), (0,))
    else:
        p_beta = tl.make_block_ptr(beta + i_b * T * H + i_h, (T,), (H,), (
            i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT, i_k * BK), (BT, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb, tl.trans(b_k), allow_tf32=False)
    b_A = -tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :],
        b_A, 0)
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BT) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]
    if HEAD_FIRST:
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT, 0), (BT, BT), (1, 0))
    else:
        p_A = tl.make_block_ptr(A + i_b * T * H * BT + i_h * BT, (T, BT), (
            H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A, boundary_check=(0, 1))
    b_A = b_A
