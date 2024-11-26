import triton
import triton.language as tl
import torch

@triton.jit
def _DWf_DW_dfg_kernel(DW, e, g, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    """
    e = e.float()
    se = 1.0 / (1.0 + torch.exp(-e))
    f = (se * e).to(dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    se_row = tl.sigmoid(e_row)
    f_row = se_row * e_row
    f_row = f_row
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    de_row = dg_row * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row
    tl.store(DW + offsets, h_row, mask=mask)
    tl.store(e + offsets, df_row, mask=mask)
    tl.store(g + offsets, de_row, mask=mask)
