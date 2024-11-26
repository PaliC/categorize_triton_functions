import triton
import triton.language as tl
import torch

@triton.jit
def load_fn(ptrs, offset_first, offset_second, _in_boundary_first,
    _in_boundary_second):
    boundary_first = _in_boundary_first
    boundary_second = _in_boundary_second
    """
    # Debugging GPU segfault
    boundary_first = 0
    boundary_second = 0
    mask = (offset_first[:, None] < boundary_first) &            (offset_second[None, :] < boundary_second)
    return tl.load(ptrs, mask=mask, other=0.0)
    """
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (offset_second[
            None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor
