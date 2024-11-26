import triton
import triton.language as tl
import torch

@triton.jit
def _block_is_filtered(check_val: 'tl.tensor', filter_eps: 'tl.tensor'
    ) ->tl.tensor:
    return tl.reduce(check_val < filter_eps, None, tl_and_reduce_fn)
