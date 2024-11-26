import triton
import triton.language as tl
import torch

@triton.jit
def fused_recurrent_rwkv4_backward_kernel(w_ptr, w_s_c, u_ptr, u_s_c, k_ptr,
    k_s_b, k_s_t, k_s_c, v_ptr, v_s_b, v_s_t, v_s_c, state_ptr, state_s_b,
    state_s_abe, state_s_t, state_s_c, gwkv_ptr, gwkv_s_b, gwkv_s_t,
    gwkv_s_c, gstate_out_ptr, gstate_out_s_b, gstate_out_s_abe,
    gstate_out_s_c, gw_ptr, gw_s_c, gu_ptr, gu_s_c, gk_ptr, gk_s_b, gk_s_t,
    gk_s_c, gv_ptr, gv_s_b, gv_s_t, gv_s_c, gstate_ptr, gstate_s_b,
    gstate_s_abe, gstate_s_c, tsz, chans, BLOCK_SIZE_C: 'tl.constexpr'):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    cs = c_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    eps_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe
    gk_ptr = gk_ptr + b_idx * gk_s_b
    gv_ptr = gv_ptr + b_idx * gv_s_b
    gwkv_ptr = gwkv_ptr + b_idx * gwkv_s_b
    galpha_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b
    gbeta_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + gstate_out_s_abe
    geps_out_ptr = (gstate_out_ptr + b_idx * gstate_out_s_b + 2 *
        gstate_out_s_abe)
    galpha = tl.load(galpha_out_ptr + gstate_out_s_c * cs, mask=cmask)
    gbeta = tl.load(gbeta_out_ptr + gstate_out_s_c * cs, mask=cmask)
    geps = tl.load(geps_out_ptr + gstate_out_s_c * cs, mask=cmask)
    w = tl.load(w_ptr + w_s_c * cs, mask=cmask)
    u = tl.load(u_ptr + u_s_c * cs, mask=cmask)
    gw = tl.zeros_like(w)
    gu = tl.zeros_like(u)
    alpha_prev = tl.load(alpha_ptr + tsz * state_s_t + state_s_c * cs, mask
        =cmask)
    beta_prev = tl.load(beta_ptr + tsz * state_s_t + state_s_c * cs, mask=cmask
        )
    eps_prev = tl.load(eps_ptr + tsz * state_s_t + state_s_c * cs, mask=cmask)
    for t in range(tsz):
        tc = tsz - t - 1
        kt = tl.load(k_ptr + tc * k_s_t + k_s_c * cs, mask=cmask)
        vt = tl.load(v_ptr + tc * v_s_t + v_s_c * cs, mask=cmask)
        alpha_curr = alpha_prev
        beta_curr = beta_prev
        eps_curr = eps_prev
        alpha_prev = tl.load(alpha_ptr + tc * state_s_t + state_s_c * cs,
            mask=cmask)
        beta_prev = tl.load(beta_ptr + tc * state_s_t + state_s_c * cs,
            mask=cmask)
        eps_prev = tl.load(eps_ptr + tc * state_s_t + state_s_c * cs, mask=
            cmask)
        ukt = u + kt
        tau = tl.maximum(ukt, eps_prev)
        e1 = tl.exp(eps_prev - tau)
        e2 = tl.exp(ukt - tau)
        euke = tl.exp(ukt + eps_prev - 2 * tau)
        denom = e1 * beta_prev + e2
        denom_sq = denom * denom
        gwkvt = tl.load(gwkv_ptr + tc * gwkv_s_t + gwkv_s_c * cs, mask=cmask)
        guk = gwkvt * e2 * (e1 * beta_prev * vt - e1 * alpha_prev) / denom_sq
        gu += guk
        gk = guk
        gv = gwkvt * e2 / denom
        galpha_wkv = gwkvt * e1 / denom
        gbeta_wkv = -gwkvt * e1 * (e2 * vt + e1 * alpha_prev) / denom_sq
        geps_wkv_denom = e1 * beta_prev + e2
        geps_wkv = gwkvt * euke * (alpha_prev - vt * beta_prev) / (
            geps_wkv_denom * geps_wkv_denom)
        e1 = tl.exp(w + eps_prev - eps_curr)
        e2 = tl.exp(kt - eps_curr)
        galpha_we = galpha * e1 * alpha_prev
        gw += galpha_we
        gk += galpha * e2 * vt
        gv += galpha * e2
        geps += galpha * -alpha_curr
        gbeta_we = gbeta * e1 * beta_prev
        gw += gbeta_we
        gk += gbeta * e2
        geps += gbeta * -beta_curr
        geps_mask = w + eps_prev > kt
        geps_we = tl.where(geps_mask, geps, tl.zeros_like(geps))
        gw += geps_we
        gk += tl.where(geps_mask, tl.zeros_like(geps), geps)
        tl.store(gk_ptr + tc * gk_s_t + gk_s_c * cs, gk, mask=cmask)
        tl.store(gv_ptr + tc * gv_s_t + gv_s_c * cs, gv, mask=cmask)
        galpha = galpha * e1 + galpha_wkv
        gbeta = gbeta * e1 + gbeta_wkv
        geps = galpha_we + gbeta_we + geps_we + geps_wkv
    galpha_ptr = gstate_ptr + b_idx * gstate_s_b
    gbeta_ptr = gstate_ptr + b_idx * gstate_s_b + gstate_s_abe
    geps_ptr = gstate_ptr + b_idx * gstate_s_b + 2 * gstate_s_abe
    tl.store(galpha_ptr + gstate_s_c * cs, galpha, mask=cmask)
    tl.store(gbeta_ptr + gstate_s_c * cs, gbeta, mask=cmask)
    tl.store(geps_ptr + gstate_s_c * cs, geps, mask=cmask)
    gw_temp = tl.load(gw_ptr + gw_s_c * cs, mask=cmask)
    gw_temp += gw
    tl.store(gw_ptr + gw_s_c * cs, gw_temp, mask=cmask)
    gu_temp = tl.load(gu_ptr + gu_s_c * cs, mask=cmask)
    gu_temp += gu
    tl.store(gu_ptr + gu_s_c * cs, gu_temp, mask=cmask)
