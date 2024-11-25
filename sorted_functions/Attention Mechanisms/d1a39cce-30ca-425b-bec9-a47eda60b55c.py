import triton
import triton.language as tl
import torch

@triton.jit
def MSAFwdFused(v_si_ptr, b_ij_ptr, g_si_ptr, output_ptr, vw_ptr,
    logsumexp_ptr, C_hidden, N_head, C_LEN_POW2: 'tl.constexpr',
    RES_LEN_POW2: 'tl.constexpr', SEQ_LEN: 'tl.constexpr', RES_LEN:
    'tl.constexpr', BLOCK_SIZE_ROW: 'tl.constexpr', BLOCK_SIZE_COL:
    'tl.constexpr', BLOCK_SIZE_SEQ: 'tl.constexpr'):
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)
    z_off = pid_z
    h_off = pid_h
    i_off = pid_i * BLOCK_SIZE_ROW
    offs_i = i_off + tl.arange(0, BLOCK_SIZE_ROW)
    offs_c = tl.arange(0, C_LEN_POW2)
    log2_e = 1.44269504089
    prev_row_max = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    new_row_max = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    l = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    for j in range(0, RES_LEN, BLOCK_SIZE_COL):
        offs_j = j + tl.arange(0, BLOCK_SIZE_COL)
        b_offs = z_off * RES_LEN * RES_LEN * N_head + offs_i[:, None
            ] * RES_LEN * N_head + offs_j[None, :] * N_head + h_off
        ij_mask = (offs_i < RES_LEN)[:, None] & (offs_j < RES_LEN)[None, :]
        b = tl.load(b_ij_ptr + b_offs, ij_mask, -INF)
        new_row_max = tl.maximum(tl.max(b, axis=1, keep_dims=True),
            prev_row_max)
        w = tl.exp2(log2_e * (b - new_row_max))
        l *= tl.exp2(log2_e * (prev_row_max - new_row_max))
        l += tl.sum(w, axis=1, keep_dims=True)
        for s in range(0, SEQ_LEN, BLOCK_SIZE_SEQ):
            for ch in range(0, C_hidden, 1):
                offs_s = s + tl.arange(0, BLOCK_SIZE_SEQ)
                si_off = (z_off * SEQ_LEN * RES_LEN * N_head * C_hidden + 
                    offs_s[None, :] * RES_LEN * N_head * C_hidden + offs_i[
                    :, None] * N_head * C_hidden + h_off * C_hidden + ch)
                sj_off = (z_off * SEQ_LEN * RES_LEN * N_head * C_hidden + 
                    offs_s[None, :] * RES_LEN * N_head * C_hidden + offs_j[
                    :, None] * N_head * C_hidden + h_off * C_hidden + ch)
                si_mask = (offs_s < SEQ_LEN)[None, :] & (offs_i < RES_LEN)[
                    :, None]
                sj_mask = (offs_s < SEQ_LEN)[None, :] & (offs_j < RES_LEN)[
                    :, None]
                v = tl.load(v_si_ptr + sj_off, sj_mask, 0)
                vw = tl.load(output_ptr + si_off, si_mask, 0)
                vw = vw * tl.exp2(log2_e * (prev_row_max - new_row_max))
                vw = tl.dot(w, v, acc=vw)
                tl.store(output_ptr + si_off, vw, si_mask)
        prev_row_max = new_row_max
    for s in range(0, SEQ_LEN, BLOCK_SIZE_SEQ):
        for ch in range(0, C_hidden, 1):
            offs_s = s + tl.arange(0, BLOCK_SIZE_SEQ)
            si_off = z_off * SEQ_LEN * RES_LEN * N_head * C_hidden + offs_s[
                None, :] * RES_LEN * N_head * C_hidden + offs_i[:, None
                ] * N_head * C_hidden + h_off * C_hidden + ch
            si_mask = (offs_s < SEQ_LEN)[None, :] & (offs_i < RES_LEN)[:, None]
            g = tl.load(g_si_ptr + si_off, si_mask, 0)
            g = tl.sigmoid(g)
            vw = tl.load(output_ptr + si_off, si_mask, 0)
            vw = vw / l
            out = g * vw
            tl.store(output_ptr + si_off, out, si_mask)
            tl.store(vw_ptr + si_off, vw, si_mask)
    lse_off = z_off * RES_LEN * N_head + offs_i[:, None] * N_head + h_off
    lse_mask = (offs_i < RES_LEN)[:, None]
    tl.store(logsumexp_ptr + lse_off, new_row_max + tl.log(l), lse_mask)
