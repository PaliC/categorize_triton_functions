import triton
import triton.language as tl
import torch

@triton.jit
def attention_fwd_kernel(q, k, v, h, o, s_qh, s_qt, s_qd, s_hh, s_ht, T,
    scale, BT: 'tl.constexpr', BD: 'tl.constexpr', NT: 'tl.constexpr',
    STORE: 'tl.constexpr', IFCOND: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    b_h = tl.zeros([BD, BD], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i *
            BT, 0), (BT, BD), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0,
            i * BT), (BD, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i *
            BT, 0), (BT, BD), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (NT * BD, BD), (s_ht, s_qd
            ), (i * BD, 0), (BD, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i *
            BT, 0), (BT, BD), (1, 0))
        if STORE:
            tl.store(p_h, b_h)
        b_q = tl.load(p_q)
        b_q = b_q * scale
        b_k = tl.load(p_k)
        b_v = tl.load(p_v)
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_o = tl.dot(b_s, b_v, allow_tf32=False)
        if IFCOND:
            if i == 0:
                b_h = tl.dot(b_k, b_v, allow_tf32=False)
            else:
                b_o += tl.dot(b_q, b_h, allow_tf32=False)
                b_h += tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
            b_h += tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o)
