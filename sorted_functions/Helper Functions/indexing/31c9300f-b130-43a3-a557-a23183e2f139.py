import triton
import triton.language as tl
import torch

@triton.jit
def _get_plane_grid_sample_locs_weights(ix, iy, ix0, iy0):
    return (ix0, iy0, ix0, iy0 + 1, ix0 + 1, iy0, ix0 + 1, iy0 + 1, ix -
        ix0, iy - iy0)
