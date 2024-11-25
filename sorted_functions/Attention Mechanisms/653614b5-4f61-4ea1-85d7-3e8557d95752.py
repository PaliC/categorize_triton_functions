import triton
import triton.language as tl
import torch

@triton.heuristics({'IS_EVEN_M': lambda args: args['N_CTX'] % args[
    'BLOCK_M'] == 0, 'IS_EVEN_N': lambda args: args['NKV_CTX'] % args[
    'BLOCK_N'] == 0})
@triton.jit
def _score_kernel(Q, K, M, sm_scale, Out, stride_qz, stride_qh, stride_qm,
    stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_oz,
    stride_oh, stride_on, Z, H, H_KV, N_CTX, ROUND_CTX, NKV_CTX,
    sliding_window_offset, sliding_window_size, SLIDING_WINDOW:
    'tl.constexpr', COMPLEMENT_SLIDING_WINDOW: 'tl.constexpr', IS_EVEN_M:
    'tl.constexpr', IS_EVEN_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // (H // H_KV)
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_hkv * stride_kh
    m_ptrs = M + off_hz * ROUND_CTX + tl.arange(0, BLOCK_M)
    o = tl.zeros([BLOCK_M], dtype=tl.float32)
    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(BLOCK_DMODEL,
        NKV_CTX), strides=(stride_kk, stride_kn), offsets=(0, start_n *
        BLOCK_N), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    if IS_EVEN_N:
        k = tl.load(K_block_ptr)
    else:
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
    lo = 0
    hi = ROUND_CTX
    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634
    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        if IS_EVEN_M:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option=
                'zero')
        m = tl.load(m_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale
        if SLIDING_WINDOW:
            dist = tl.arange(0, BLOCK_M)[:, None] - tl.arange(0, BLOCK_N)[
                None, :] + start_m - start_n * BLOCK_N + sliding_window_offset
            if COMPLEMENT_SLIDING_WINDOW:
                mask = dist >= sliding_window_size
            else:
                mask = (dist >= 0) & (dist < sliding_window_size)
        qk = qk - m[:, None]
        p = tl.math.exp2(qk)
        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)
        if not IS_EVEN_N:
            p = tl.where((tl.arange(0, BLOCK_M) + start_m < N_CTX)[:, None],
                p, 0)
        o += tl.sum(p, axis=0)
        Q_block_ptr = tl.advance(Q_block_ptr, offsets=(BLOCK_M, 0))
        m_ptrs = m_ptrs + BLOCK_M
    o_offset = off_z * stride_oz + off_h * stride_oh
    o_range = tl.arange(0, BLOCK_N) + start_n * BLOCK_N
    o_ptrs = Out + o_offset + o_range
    tl.store(o_ptrs, o, mask=o_range < NKV_CTX)
