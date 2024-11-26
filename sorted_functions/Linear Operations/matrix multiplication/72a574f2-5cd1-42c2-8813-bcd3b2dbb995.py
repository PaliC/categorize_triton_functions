import triton
import triton.language as tl
import torch

@triton.jit
def fwd_recurrence(A, B, C, Dt, X, Y, H, start, initial_state, T:
    'tl.constexpr', D: 'tl.constexpr', K: 'tl.constexpr', BV: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    dt_ptr = Dt + i_bh * T * D + i_v * BV + tl.arange(0, BV)
    u_ptr = X + i_bh * T * D + i_v * BV + tl.arange(0, BV)
    o_ptr = Y + i_bh * T * D + i_v * BV + tl.arange(0, BV)
    start_ptr = start + i_bh * T
    h = tl.zeros([BV, K], dtype=tl.float32)
    b_ptr = B + i_bh * T * K + tl.arange(0, K)
    A = A + (i_v * BV + tl.arange(0, BV)[:, None]) * K + tl.arange(0, K)[
        None, :]
    _A = tl.load(A)
    H_ptr = H + i_bh * T * D * K + (i_v * BV + tl.arange(0, BV)[:, None]
        ) * K + tl.arange(0, K)[None, :]
    h += tl.load(initial_state + i_bh * D * K + (i_v * BV + tl.arange(0, BV
        )[:, None]) * K + tl.arange(0, K)[None, :])
    for i in range(T):
        b = tl.load(b_ptr)
        dt = tl.load(dt_ptr)
        start_flag = tl.load(start_ptr)
        u = tl.load(u_ptr)
        x_dt = u * dt
        x_dt_b = x_dt[:, None] * b[None, :]
        dt_a = tl.exp(dt[:, None] * _A) * (1 - start_flag)
        h = h * dt_a + x_dt_b
        tl.store(H_ptr, h)
        b_ptr += K
        dt_ptr += D
        start_ptr += 1
        u_ptr += D
        o_ptr += D
        H_ptr += D * K
