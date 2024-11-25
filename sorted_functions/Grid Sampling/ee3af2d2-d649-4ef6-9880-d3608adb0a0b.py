import triton
import triton.language as tl
import torch

@triton.jit
def _sample_2d(image, w, batch_index, ix, iy, IH, IW, C: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr'):
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1)
    image_offs = image + batch_index * IW * IH * C + iy_ * IW * C + ix_ * C
    mask_w = w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW))
    if C == 1:
        val = tl.view(tl.load(image_offs), (BLOCK_SIZE,))
        out = tl.view(val * mask_w, (BLOCK_SIZE,))
        return out
    else:
        val = tl.view(tl.load(image_offs[:, None] + Coffs[None, :]), (
            BLOCK_SIZE, C))
        mask_w_bcast = tl.view(mask_w[:, None], (BLOCK_SIZE, 1))
        return val * mask_w_bcast
