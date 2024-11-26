import triton
import triton.language as tl
import torch

@triton.jit
def _approx_backward_kernel(DW, e, g, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    """
    f = 1/2 * e * (1 + tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) ))
    h = f * up

    df/de (with help from https://arxiv.org/pdf/2305.12073.pdf :))
    df/de = 1/2 * [1 + tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) )] +
            1/2 * sech^2 [   sqrt(2/pi) * x * (1 + 0.044715 * x^2 )  ] *                            ( sqrt(2/pi) * x * (1 + 0.044715 * x^2 * 3 ) )

    Notice sech^2(x) = 1 - tanh^2(x)
    So reuse tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) )

    See https://www.desmos.com/calculator/nqprfoni6x
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    s = 0.7978845608028654
    a = s * e_row
    b = a * 0.044715 * e_row * e_row
    T = 1.0 + triton_tanh(a + b)
    T2 = 0.5 * T
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
    df_de = T2 + Q2
    f_row = T2 * e_row
    f_row = f_row
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    de_row = dg_row * df_de
    de_row = de_row
    tl.store(DW + offsets, h_row, mask=mask)
    tl.store(e + offsets, df_row, mask=mask)
    tl.store(g + offsets, de_row, mask=mask)
