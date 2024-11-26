import triton
import triton.language as tl
import torch

@triton.jit
def triton_cross_merge_unidi(x, y, BC: 'tl.constexpr', BH: 'tl.constexpr',
    BW: 'tl.constexpr', DC: 'tl.constexpr', DH: 'tl.constexpr', DW:
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
    p_y2 = y + i_b * 4 * _tmp1 + _tmp1 + _tmp2
    p_y3 = y + i_b * 4 * _tmp1 + 2 * _tmp1 + _tmp2
    p_y4 = y + i_b * 4 * _tmp1 + 3 * _tmp1 + _tmp2
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _y1 = tl.load(p_y1 + _idx, mask=_mask_hw)
        _y2 = tl.load(p_y2 + _idx, mask=_mask_hw)
        _y3 = tl.load(p_y3 + _idx, mask=_mask_hw)
        _y4 = tl.load(p_y4 + _idx, mask=_mask_hw)
        tl.store(p_x + _idx, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)
