import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16)], key=['BK'])
@triton.jit
def fwd_prepare_wy_repr_kernel_chunk64(k, beta, A, T: 'tl.constexpr', H:
    'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BK:
    'tl.constexpr', BC: 'tl.constexpr', HEAD_FIRST: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_A2 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A3 = tl.zeros([BC, BC], dtype=tl.float32)
    if HEAD_FIRST:
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,),
            (BC,), (0,))
    else:
        p_beta = tl.make_block_ptr(beta + i_b * T * H + i_h, (T,), (H,), (
            i_t * BT,), (BC,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    if HEAD_FIRST:
        p_beta2 = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT +
            BC,), (BC,), (0,))
    else:
        p_beta2 = tl.make_block_ptr(beta + i_b * T * H + i_h, (T,), (H,), (
            i_t * BT + BC,), (BC,), (0,))
    b_beta2 = tl.load(p_beta2, boundary_check=(0,))
    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t *
                BT + BC, i_k * BK), (BC, BK), (1, 0))
        else:
            p_k = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k + i_b * T * H * K + i_h * K, (T, K),
                (H * K, 1), (i_t * BT + BC, i_k * BK), (BC, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_kb2 = b_k2 * b_beta2[:, None]
        b_A += tl.dot(b_kb, tl.trans(b_k), allow_tf32=False)
        b_A2 += tl.dot(b_kb2, tl.trans(b_k2), allow_tf32=False)
        b_A3 += tl.dot(b_kb2, tl.trans(b_k), allow_tf32=False)
    b_A = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :],
        b_A, 0)
    b_A2 = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :],
        b_A2, 0)
    for i in range(1, BC):
        mask = tl.arange(0, BC) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BC) < i)
        b_a2 = b_a2 + tl.sum(b_a2[:, None] * b_A2, 0) * (tl.arange(0, BC) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
        b_A2 = tl.where(mask[:, None], b_a2, b_A2)
    b_A += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A3 = -tl.dot(tl.dot(b_A2, b_A3, allow_tf32=False), b_A, allow_tf32=False)
    if HEAD_FIRST:
        p_A1 = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT, 0), (BC, BC), (1, 0))
        p_A2 = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT + BC, BC), (BC, BC), (1, 0))
        p_A3 = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT + BC, 0), (BC, BC), (1, 0))
        p_A4 = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t *
            BT, BC), (BC, BC), (1, 0))
    else:
        p_A1 = tl.make_block_ptr(A + i_b * T * H * BT + i_h * BT, (T, BT),
            (H * BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
        p_A2 = tl.make_block_ptr(A + i_b * T * H * BT + i_h * BT, (T, BT),
            (H * BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
        p_A3 = tl.make_block_ptr(A + i_b * T * H * BT + i_h * BT, (T, BT),
            (H * BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
        p_A4 = tl.make_block_ptr(A + i_b * T * H * BT + i_h * BT, (T, BT),
            (H * BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))
    tl.store(p_A1, b_A, boundary_check=(0, 1))
    tl.store(p_A2, b_A2, boundary_check=(0, 1))
    tl.store(p_A3, b_A3, boundary_check=(0, 1))
    tl.store(p_A4, tl.zeros([BC, BC], dtype=tl.float32), boundary_check=(0, 1))
