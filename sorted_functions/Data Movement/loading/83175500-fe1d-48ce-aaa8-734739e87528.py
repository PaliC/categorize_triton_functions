import triton
import triton.language as tl
import torch

@triton.jit
def load_2d(ptr, sz0: 'const', sz1: 'const', n0, n1, max0, max1, stride0=
    None, stride1=1):
    """Chunk 2d matrix (defined by ptr) into 2d grid, where each chunk has size (sz0,sz1). Load the (n0,n1)th chunk. Ie, load [n0*sz0,...,(n0+1)*sz0-1] x [n1*sz1,...,(n1+1)*sz1-1]."""
    stride0 = stride0 or sz1
    offs0 = offset_1d(sz0, n0)
    offs1 = offset_1d(sz1, n1)
    offs = offset_2d(offs0, offs1, stride0, stride1)
    mask = mask_2d(offs0, offs1, max0, max1)
    return tl.load(ptr + offs, mask)
