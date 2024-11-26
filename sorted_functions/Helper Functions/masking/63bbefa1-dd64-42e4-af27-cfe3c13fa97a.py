import triton
import triton.language as tl
import torch

@triton.jit
def debug_fill_dropout_rng_tensor(R, stride_rz, stride_rh, stride_rm,
    stride_rn, seqlen_q, seqlen_k, philox_seed_ptr, philox_offset_base_ptr,
    BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    philox_seed = tl.load(philox_seed_ptr)
    philox_offset_base = tl.load(philox_offset_base_ptr)
    debug_fill_dropout_rng(R, stride_rz, stride_rh, stride_rm, stride_rn,
        seqlen_q, seqlen_k, philox_seed, philox_offset_base, BLOCK_M, BLOCK_N)
