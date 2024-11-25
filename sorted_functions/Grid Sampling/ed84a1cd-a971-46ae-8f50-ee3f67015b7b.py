import triton
import triton.language as tl
import torch

@triton.jit
def _splat_3d(to_splat, grad_image, w, batch_index, ix, iy, iz, ID:
    'tl.constexpr', IH: 'tl.constexpr', IW: 'tl.constexpr', C:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    channel_bcast = tl.full((1, C), 1.0, dtype=tl.float32)
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1)
    iz_ = tl.minimum(tl.maximum(iz, 0.0), ID - 1)
    w = tl.view(w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW) * (iz <
        ID) * (iz >= 0))[:, None] * channel_bcast, (BLOCK_SIZE, C))
    offs = tl.view((batch_index * ID * IW * IH * C + iz_ * IW * IH * C + 
        iy_ * IW * C + ix_ * C)[:, None] + Coffs[None, :], (BLOCK_SIZE, C))
    tl.atomic_add(grad_image + offs, w * to_splat)
