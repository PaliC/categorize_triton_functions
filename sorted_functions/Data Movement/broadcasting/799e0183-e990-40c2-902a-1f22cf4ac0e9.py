import triton
import triton.language as tl
import torch

@triton.jit
def split_2D_jagged_jagged_w_prefix(JaggedIn, OffsetsA, OffsetsB, OutA,
    OutB, D, stride_id, stride_ad, stride_bd, n_prefix_to_B, BLOCK_D:
    'tl.constexpr'):
    split_2D_jagged_w_prefix(JaggedIn, 0, OffsetsA, OffsetsB, OutA, OutB, D,
        stride_id, stride_ad, stride_bd, n_prefix_to_B, IS_DENSE_A=False,
        IS_DENSE_B=False, BLOCK_D=BLOCK_D, IS_REPLACE=False)
