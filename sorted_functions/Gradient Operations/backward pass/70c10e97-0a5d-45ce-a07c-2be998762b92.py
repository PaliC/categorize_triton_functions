import triton
import triton.language as tl
import torch

@triton.jit
def _exact_backward_kernel(DW, e, g, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    """
    f = 1/2 * e * (1 + erf(1/sqrt(2) * e))
    h = f * up

    df/de (with help of Wolfram :)
    df/de = 1/2 * (1 + erf(1/sqrt(2) * e)) + 1/sqrt(2*pi) * e * exp(-1/2 * e^2)

    Reuse via
    f =        1/2 * (1 + erf(1/sqrt(2) * e)) * e
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    f_partial_row = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_partial_row * e_row
    f_row = f_row
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    t = 0.3989422804014327
    df_de = f_partial_row + t * e_row * tl.exp(-0.5 * e_row * e_row)
    de_row = dg_row * df_de
    de_row = de_row
    tl.store(DW + offsets, h_row, mask=mask)
    tl.store(e + offsets, df_row, mask=mask)
    tl.store(g + offsets, de_row, mask=mask)
