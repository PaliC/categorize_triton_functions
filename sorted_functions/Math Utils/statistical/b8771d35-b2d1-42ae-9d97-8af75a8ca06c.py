import triton
import triton.language as tl
import torch

@triton.jit
def kl_div_kernel(logits, target_logits, loss, s_logits, s_loss, reduction:
    'tl.constexpr', N: 'tl.constexpr', V: 'tl.constexpr', BV: 'tl.constexpr'):
    i_n = tl.program_id(0)
    logits += i_n * s_logits
    target_logits += i_n * s_logits
    sm, tm = float('-inf'), float('-inf')
    sd, td = 0.0, 0.0
    NV = tl.cdiv(V, BV)
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float('-inf'))
        b_sm = tl.max(b_sl)
        m_new = tl.maximum(sm, b_sm)
        sd = sd * tl.exp(sm - m_new) + tl.sum(tl.exp(b_sl - m_new))
        sm = m_new
        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float('-inf'))
        b_tm = tl.max(b_tl)
        m_new = tl.maximum(tm, b_tm)
        td = td * tl.exp(tm - m_new) + tl.sum(tl.exp(b_tl - m_new))
        tm = m_new
    b_loss = 0.0
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float('-inf'))
        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float('-inf'))
        b_sp_log = b_sl - sm - tl.log(sd)
        b_tp_log = b_tl - tm - tl.log(td)
        b_sp = tl.exp(b_sp_log)
        b_tp = tl.exp(b_tp_log)
        b_kl = tl.where(o_x < V, b_tp * (b_tp_log - b_sp_log), 0)
        b_dl = -b_tp + b_sp
        b_loss += tl.sum(b_kl)
        if reduction == 'batchmean':
            b_dl = b_dl / N
        tl.store(logits + o_x, b_dl, mask=o_x < V)
    if reduction == 'batchmean':
        b_loss = b_loss / N
    tl.store(loss + i_n * s_loss, b_loss)
