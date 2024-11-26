import triton
import triton.language as tl
import torch

@triton.jit
def concat_2D_jagged_jagged_w_prefix(OffsetsA, ValuesA, OffsetsB, ValuesB,
    Out, D, stride_ad, stride_bd, stride_od, n_prefix_from_B, BLOCK_D:
    'tl.constexpr'):
    concat_2D_jagged_w_prefix(OffsetsA, ValuesA, OffsetsB, ValuesB, 0, Out,
        D, stride_ad, stride_bd, 0, stride_od, n_prefix_from_B, IS_DENSE_A=
        False, IS_DENSE_B=False, BLOCK_D=BLOCK_D, IS_REPLACE=False)
