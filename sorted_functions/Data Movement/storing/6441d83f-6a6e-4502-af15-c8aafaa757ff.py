import triton
import triton.language as tl
import torch

@triton.jit
def _splat_2d(to_splat, grad_image, w, batch_index, ix, iy, IH, IW, C:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1)
    w = w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW))
    w = tl.view(w[:, None], (BLOCK_SIZE, 1))
    offs = tl.view((batch_index * IW * IH * C + iy_ * IW * C + ix_ * C)[:,
        None] + Coffs[None, :], (BLOCK_SIZE, C))
    tl.atomic_add(grad_image + offs, w * to_splat)
