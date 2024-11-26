import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': block_size},
    num_warps=num_warps) for block_size, num_warps in itertools.product([32,
    64, 128, 256, 512, 1024, 2048, 4096], [1, 2, 4, 8, 16, 32])], key=[
    'length', 'kernel_size', 'stride', 'n_frames'])
@triton.jit
def unfold_kernel(input_ptr, output_ptr, length, kernel_size, stride,
    n_frames, BLOCK_SIZE: 'tl.constexpr'):
    batch_idx = tl.program_id(0)
    frame_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = frame_idx < n_frames
    input_pos = frame_idx * stride
    for i in range(kernel_size):
        in_bounds = mask & (input_pos + i < length)
        val = tl.where(in_bounds, tl.load(input_ptr + batch_idx * length +
            input_pos + i, mask=in_bounds), 0)
        out_idx = (batch_idx * n_frames * kernel_size + frame_idx *
            kernel_size + i)
        tl.store(output_ptr + out_idx, val, mask=in_bounds)
