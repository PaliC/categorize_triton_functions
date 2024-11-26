import triton
import triton.language as tl
import torch

@triton.jit
def _voxel_grid_splat(to_splat, grad_image, batch_index, ix, iy, iz, N:
    'tl.constexpr', C: 'tl.constexpr', ID: 'tl.constexpr', IH:
    'tl.constexpr', IW: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    ix = (ix + 1) / 2 * IW - 0.5
    iy = (iy + 1) / 2 * IH - 0.5
    iz = (iz + 1) / 2 * ID - 0.5
    ix0 = ix - ix % 1
    iy0 = iy - iy % 1
    iz0 = iz - iz % 1
    V000x = ix0
    V000y = iy0
    V000z = iz0
    V100x = ix0
    V100y = iy0
    V100z = iz0 + 1
    V010x = ix0
    V010y = iy0 + 1
    V010z = iz0
    V001x = ix0 + 1
    V001y = iy0
    V001z = iz0
    V101x = ix0 + 1
    V101y = iy0
    V101z = iz0 + 1
    V011x = ix0 + 1
    V011y = iy0 + 1
    V011z = iz0
    V110x = ix0
    V110y = iy0 + 1
    V110z = iz0 + 1
    V111x = ix0 + 1
    V111y = iy0 + 1
    V111z = iz0 + 1
    x = ix - ix0
    y = iy - iy0
    z = iz - iz0
    _splat_3d(to_splat, grad_image, (1 - x) * (1 - y) * (1 - z),
        batch_index, V000x, V000y, V000z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_image, (1 - x) * (1 - y) * z, batch_index,
        V100x, V100y, V100z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_image, (1 - x) * y * (1 - z), batch_index,
        V010x, V010y, V010z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_image, x * (1 - y) * (1 - z), batch_index,
        V001x, V001y, V001z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_image, x * (1 - y) * z, batch_index, V101x,
        V101y, V101z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_image, x * y * (1 - z), batch_index, V011x,
        V011y, V011z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_image, (1 - x) * y * z, batch_index, V110x,
        V110y, V110z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_image, x * y * z, batch_index, V111x, V111y,
        V111z, ID, IH, IW, C, BLOCK_SIZE)
