import triton
import triton.language as tl
import torch

@triton.jit
def _grid_splat(to_splat, grad_image, batch_index, ix, iy, N:
    'tl.constexpr', C: 'tl.constexpr', IH: 'tl.constexpr', IW:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    ix = (ix + 1) / 2 * IW - 0.5
    iy = (iy + 1) / 2 * IH - 0.5
    ix_nw = ix - ix % 1
    iy_nw = iy - iy % 1
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1
    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)
    _splat_2d(to_splat, grad_image, nw, batch_index, ix_nw, iy_nw, IH, IW,
        C, BLOCK_SIZE)
    _splat_2d(to_splat, grad_image, ne, batch_index, ix_ne, iy_ne, IH, IW,
        C, BLOCK_SIZE)
    _splat_2d(to_splat, grad_image, sw, batch_index, ix_sw, iy_sw, IH, IW,
        C, BLOCK_SIZE)
    _splat_2d(to_splat, grad_image, se, batch_index, ix_se, iy_se, IH, IW,
        C, BLOCK_SIZE)
