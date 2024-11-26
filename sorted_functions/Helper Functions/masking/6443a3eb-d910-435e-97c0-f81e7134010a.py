import triton
import triton.language as tl
import torch

@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE:
    'tl.constexpr'):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 4
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    r0, r1, r2, r3 = tl.random.rand4x(seed, offset)
    scale = 1.0 / (1.0 - p)
    for i in tl.static_range(4):
        curr_offset = offset + BLOCK_SIZE * i
        mask = curr_offset < n_elements
        x = tl.load(x_ptr + curr_offset, mask=mask)
        r = tl.where(i == 0, r0, tl.where(i == 1, r1, tl.where(i == 2, r2, r3))
            )
        keep = r > p
        output = tl.where(keep, x * scale, 0.0)
        tl.store(output_ptr + curr_offset, output, mask=mask)
