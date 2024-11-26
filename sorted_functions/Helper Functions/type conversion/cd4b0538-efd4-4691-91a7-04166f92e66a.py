import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 128,
    'PACKED_BLOCK_SIZE': 16}), triton.Config({'BLOCK_SIZE': 256,
    'PACKED_BLOCK_SIZE': 32}), triton.Config({'BLOCK_SIZE': 512,
    'PACKED_BLOCK_SIZE': 64}), triton.Config({'BLOCK_SIZE': 1024,
    'PACKED_BLOCK_SIZE': 128}), triton.Config({'BLOCK_SIZE': 2048,
    'PACKED_BLOCK_SIZE': 256})], key=['n_elements'])
@triton.jit
def unpack_sign_kernel(packed_ptr, unpacked_ptr, n_elements, BLOCK_SIZE:
    'tl.constexpr', PACKED_BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    packed_start = pid * PACKED_BLOCK_SIZE
    packed_offsets = packed_start + tl.arange(0, PACKED_BLOCK_SIZE)
    packed_mask = packed_offsets < (n_elements + 7) // 8
    packed = tl.load(packed_ptr + packed_offsets, mask=packed_mask, other=0)
    bit_offsets = tl.arange(0, 8)
    packed = packed[:, None]
    bits = packed >> 7 - bit_offsets & 1
    signs = bits * 2 - 1
    element_offsets = block_start + (tl.arange(0, PACKED_BLOCK_SIZE)[:,
        None] * 8 + bit_offsets)
    element_offsets = tl.reshape(element_offsets, [BLOCK_SIZE])
    signs = tl.reshape(signs, [BLOCK_SIZE])
    mask = element_offsets < n_elements
    tl.store(unpacked_ptr + element_offsets, signs, mask=mask)
