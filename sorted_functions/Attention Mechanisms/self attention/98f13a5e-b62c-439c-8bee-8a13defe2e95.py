import triton
import triton.language as tl
import torch

@triton.jit
def parallel_rebased_fwd_kernel(q, k, v, o, z, s_k_h, s_k_t, s_k_d, s_v_h,
    s_v_t, s_v_d, scale, B, H, T, K: 'tl.constexpr', V: 'tl.constexpr', BTL:
    'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'
    ):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_c *
        BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k *
        BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (0, 
        i_v * BV), (BTS, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)
        b_o = b_o + tl.dot(b_s, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k *
        BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_c *
        BTL, i_v * BV), (BTS, BV), (1, 0))
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_v_h, (T, V), (
        s_v_t, s_v_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o, boundary_check=(0, 1))
    tl.store(p_z, b_z, mask=i_c * BTL + tl.arange(0, BTL) < T)
