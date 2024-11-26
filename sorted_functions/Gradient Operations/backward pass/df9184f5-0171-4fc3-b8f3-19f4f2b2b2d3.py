import triton
import triton.language as tl
import torch

@triton.jit
def bwd_recurrence(A, B, C, U, Dt, DO, H, start, DA, DB, DC, dDt, dU, batch,
    initial_state, grad_detach, T: 'tl.constexpr', D: 'tl.constexpr', K:
    'tl.constexpr', BV: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    NV = tl.cdiv(D, BV)
    dt_ptr = Dt + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    ddt_ptr = dDt + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    u_ptr = U + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    du_ptr = dU + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    do_ptr = DO + i_bh * T * D + i_v * BV + tl.arange(0, BV) + (T - 1) * D
    start_ptr = start + i_bh * T + (T - 1)
    grad_detach_ptr = grad_detach + i_bh * T + (T - 1)
    dh = tl.zeros([BV, K], dtype=tl.float32)
    dA = tl.zeros([BV, K], dtype=tl.float32)
    b_ptr = B + i_bh * T * K + tl.arange(0, K) + (T - 1) * K
    c_ptr = C + i_bh * T * K + tl.arange(0, K) + (T - 1) * K
    dc_ptr = DC + (i_bh + batch * i_v) * T * K + tl.arange(0, K) + (T - 1) * K
    db_ptr = DB + (i_bh + batch * i_v) * T * K + tl.arange(0, K) + (T - 1) * K
    A = A + (i_v * BV + tl.arange(0, BV)[:, None]) * K + tl.arange(0, K)[
        None, :]
    _A = tl.load(A)
    H_ptr = H + i_bh * T * D * K + (i_v * BV + tl.arange(0, BV)[:, None]
        ) * K + tl.arange(0, K)[None, :] + (T - 1) * D * K
    for i in range(T):
        h = tl.load(H_ptr)
        if i < T - 1:
            next_h = tl.load(H_ptr - D * K)
        else:
            next_h = tl.load(initial_state + i_bh * D * K + (i_v * BV + tl.
                arange(0, BV)[:, None]) * K + tl.arange(0, K)[None, :])
        b = tl.load(b_ptr)
        c = tl.load(c_ptr)
        do = tl.load(do_ptr)
        u = tl.load(u_ptr)
        dt = tl.load(dt_ptr)
        start_flag = tl.load(start_ptr)
        grad_detach_flag = tl.load(grad_detach_ptr)
        dh = dh * (1 - grad_detach_flag)
        dc = tl.sum(h * do[:, None], axis=0)
        tl.store(dc_ptr, dc)
        dh += do[:, None] * c[None, :]
        dt_u = dt * u
        db = tl.sum(dh * dt_u[:, None], axis=0)
        tl.store(db_ptr, db)
        ddt_u = tl.sum(dh * b[None, :], axis=1)
        ddt = ddt_u * u
        du = ddt_u * dt
        tl.store(du_ptr, du)
        dt_a = tl.exp(dt[:, None] * _A) * (1 - start_flag)
        dh *= dt_a
        d_decay = dh * next_h
        dA += d_decay * dt[:, None]
        ddt += tl.sum(d_decay * _A, axis=1)
        tl.store(ddt_ptr, ddt)
        b_ptr -= K
        c_ptr -= K
        dc_ptr -= K
        db_ptr -= K
        dt_ptr -= D
        ddt_ptr -= D
        u_ptr -= D
        du_ptr -= D
        do_ptr -= D
        H_ptr -= D * K
        start_ptr -= 1
        grad_detach_ptr -= 1
    DA_ptr = DA + i_bh * D * K + (i_v * BV + tl.arange(0, BV)[:, None]
        ) * K + tl.arange(0, K)[None, :]
    tl.store(DA_ptr, dA)
