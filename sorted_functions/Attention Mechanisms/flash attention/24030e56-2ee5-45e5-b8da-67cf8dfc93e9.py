import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_rwkv4_forward_kernel(w_ptr, w_s_c, u_ptr, u_s_c, k_ptr,
    k_s_b, k_s_t, k_s_c, v_ptr, v_s_b, v_s_t, v_s_c, state_ptr, state_s_b,
    state_s_abe, state_s_c, wkv_ptr, wkv_s_b, wkv_s_t, wkv_s_c,
    state_out_ptr, state_out_s_b, state_out_s_abe, state_out_s_t,
    state_out_s_c, chans, tsz, BLOCK_SIZE_C: 'tl.constexpr'):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    cs = c_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    eps_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe
    wkv_ptr = wkv_ptr + b_idx * wkv_s_b
    alpha_out_ptr = state_out_ptr + b_idx * state_out_s_b
    beta_out_ptr = state_out_ptr + b_idx * state_out_s_b + state_out_s_abe
    eps_out_ptr = state_out_ptr + b_idx * state_out_s_b + 2 * state_out_s_abe
    alpha = tl.load(alpha_ptr + cs * state_s_c, mask=cmask)
    beta = tl.load(beta_ptr + cs * state_s_c, mask=cmask)
    eps = tl.load(eps_ptr + cs * state_s_c, mask=cmask)
    w = tl.load(w_ptr + cs * w_s_c, mask=cmask)
    u = tl.load(u_ptr + cs * u_s_c, mask=cmask)
    for t in range(tsz):
        kt = tl.load(k_ptr + t * k_s_t + cs * k_s_c, mask=cmask)
        vt = tl.load(v_ptr + t * v_s_t + cs * v_s_c, mask=cmask)
        ukt = u + kt
        tau = tl.maximum(ukt, eps)
        e1a = tl.exp(eps - tau)
        e2a = tl.exp(ukt - tau)
        wkv = (e1a * alpha + e2a * vt) / (e1a * beta + e2a)
        tl.store(wkv_ptr + t * wkv_s_t + cs * wkv_s_c, wkv, mask=cmask)
        w_eps = w + eps
        eps = tl.maximum(w_eps, kt)
        e1b = tl.exp(w_eps - eps)
        e2b = tl.exp(kt - eps)
        alpha = e1b * alpha + e2b * vt
        beta = e1b * beta + e2b
        tl.store(alpha_out_ptr + t * state_out_s_t + cs * state_out_s_c,
            alpha, mask=cmask)
        tl.store(beta_out_ptr + t * state_out_s_t + cs * state_out_s_c,
            beta, mask=cmask)
        tl.store(eps_out_ptr + t * state_out_s_t + cs * state_out_s_c, eps,
            mask=cmask)
