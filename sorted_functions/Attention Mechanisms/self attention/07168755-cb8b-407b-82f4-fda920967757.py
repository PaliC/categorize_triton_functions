import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m,
    qk_scale, N_CTX, sliding_window_offset, sliding_window_size, BLOCK_M:
    'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    SLIDING_WINDOW: 'tl.constexpr', IS_EVEN_M: 'tl.constexpr', IS_EVEN_N:
    'tl.constexpr', COMPLEMENT_SLIDING_WINDOW: 'tl.constexpr'):
    if SLIDING_WINDOW and not COMPLEMENT_SLIDING_WINDOW:
        if COMPLEMENT_SLIDING_WINDOW:
            lo = 0
            hi = ((start_m + 1) * BLOCK_M + sliding_window_offset -
                sliding_window_size + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        else:
            lo = (start_m * BLOCK_M + sliding_window_offset -
                sliding_window_size + 1) // BLOCK_N * BLOCK_N
            hi = ((start_m + 1) * BLOCK_M - 1 + sliding_window_offset + BLOCK_N
                ) // BLOCK_N * BLOCK_N
            if lo < 0:
                lo = 0
            if hi > N_CTX:
                hi = N_CTX
            lo = tl.multiple_of(lo, BLOCK_N)
            K_block_ptr = tl.advance(K_block_ptr, (0, lo))
            V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    else:
        lo, hi = 0, N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if IS_EVEN_N:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option=
                'zero')
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale
        if SLIDING_WINDOW:
            dist = tl.arange(0, BLOCK_M)[:, None] - tl.arange(0, BLOCK_N)[
                None, :] + start_m * BLOCK_M - start_n + sliding_window_offset
            if COMPLEMENT_SLIDING_WINDOW:
                mask = dist >= sliding_window_size
            else:
                mask = (dist >= 0) & (dist < sliding_window_size)
            qk = tl.where(mask, qk, float('-inf'))
        if not IS_EVEN_N:
            qk = tl.where((tl.arange(0, BLOCK_N) + start_n < N_CTX)[None, :
                ], qk, float('-inf'))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)
        if not IS_EVEN_N:
            p = tl.where((tl.arange(0, BLOCK_N) + start_n < N_CTX)[None, :],
                p, 0)
        l_ij = tl.sum(p, 1)
        tmp = m_i - m_ij
        alpha_mask = tmp != tmp
        alpha = tl.math.exp2(tmp)
        alpha = tl.where(alpha_mask, 1.0, alpha)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        if IS_EVEN_N:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option=
                'zero')
        acc += tl.dot(p, v)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i
