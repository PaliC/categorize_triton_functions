import triton
import triton.language as tl
import torch

@triton.jit
def _load_mlp_bias_bcast(biases, C, offs, BLOCK_SIZE):
    return tl.view(tl.load((biases + offs + tl.arange(0, C))[None, :] + tl.
        zeros((BLOCK_SIZE, 1), dtype=tl.int32)), (BLOCK_SIZE, C))
