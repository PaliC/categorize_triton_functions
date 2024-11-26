import triton
import triton.language as tl
import torch

@triton.jit
def _get_voxel_grid_sample_locs_weights(ix, iy, iz, ix0, iy0, iz0):
    return (ix0, iy0, iz0, ix0, iy0, iz0 + 1, ix0, iy0 + 1, iz0, ix0 + 1,
        iy0, iz0, ix0 + 1, iy0, iz0 + 1, ix0 + 1, iy0 + 1, iz0, ix0, iy0 + 
        1, iz0 + 1, ix0 + 1, iy0 + 1, iz0 + 1, ix - ix0, iy - iy0, iz - iz0)
