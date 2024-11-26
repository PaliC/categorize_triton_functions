import triton
import triton.language as tl
import torch

@triton.jit
def triton_cross_scan(x, y, BC: 'tl.constexpr', BH: 'tl.constexpr', BW:
    'tl.constexpr', DC: 'tl.constexpr', DH: 'tl.constexpr', DW:
    'tl.constexpr', NH: 'tl.constexpr', NW: 'tl.constexpr'):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = i_hw // NW, i_hw % NW
    _mask_h = i_h * BH + tl.arange(0, BH) < DH
    _mask_w = i_w * BW + tl.arange(0, BW) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)
    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = _tmp0 + i_h * BH * DW + tl.arange(0, BH)[:, None
        ] * DW + i_w * BW + tl.arange(0, BW)[None, :]
    p_x = x + i_b * _tmp1 + _tmp2
    p_y1 = y + i_b * 4 * _tmp1 + _tmp2
    p_y2 = y + i_b * 4 * _tmp1 + _tmp1 + _tmp0 + i_w * BW * DH + tl.arange(
        0, BW)[None, :] * DH + i_h * BH + tl.arange(0, BH)[:, None]
    p_y3 = y + i_b * 4 * _tmp1 + 2 * _tmp1 + _tmp0 + (NH - i_h - 1
        ) * BH * DW + (BH - 1 - tl.arange(0, BH)[:, None]) * DW + (NW - i_w - 1
        ) * BW + (BW - 1 - tl.arange(0, BW)[None, :]) + (DH - NH * BH) * DW + (
        DW - NW * BW)
    p_y4 = y + i_b * 4 * _tmp1 + 3 * _tmp1 + _tmp0 + (NW - i_w - 1
        ) * BW * DH + (BW - 1 - tl.arange(0, BW)[None, :]) * DH + (NH - i_h - 1
        ) * BH + (BH - 1 - tl.arange(0, BH)[:, None]) + (DH - NH * BH) + (DW -
        NW * BW) * DH
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y1 + _idx, _x, mask=_mask_hw)
        tl.store(p_y2 + _idx, _x, mask=_mask_hw)
        tl.store(p_y3 + _idx, _x, mask=_mask_hw)
        tl.store(p_y4 + _idx, _x, mask=_mask_hw)
