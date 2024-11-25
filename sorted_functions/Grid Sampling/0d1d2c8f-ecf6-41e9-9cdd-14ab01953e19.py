import triton
import triton.language as tl
import torch

@triton.jit
def triton_local_scan(x, y, K: 'tl.constexpr', flip: 'tl.constexpr', BC:
    'tl.constexpr', BH: 'tl.constexpr', BW: 'tl.constexpr', DC:
    'tl.constexpr', DH: 'tl.constexpr', DW: 'tl.constexpr', NH:
    'tl.constexpr', NW: 'tl.constexpr'):
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
    _i = (tl.arange(0, BH) + BH * i_h)[:, None]
    _j = (tl.arange(0, BW) + BW * i_w)[None, :]
    _c_offset = (DW // K * (_i // K) + _j // K) * K * K + _i % K * K + _j % K
    if flip:
        _c_offset = DH * DW - _c_offset - 1
    p_y = y + i_b * _tmp1 + _tmp0 + _c_offset
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()
