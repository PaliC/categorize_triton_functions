import triton
import triton.language as tl
import torch

@triton.jit
def argmax_kernel(output_ptr, input_ptr, num_batches, size, block_size:
    'tl.constexpr'):
    batch = tl.program_id(0)
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(num_batches,),
        strides=(1,), offsets=(batch,), block_shape=(1,), order=(0,))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(num_batches, size
        ), strides=(size, 1), offsets=(batch, 0), block_shape=(1,
        block_size), order=(1, 0))
    input = tl.load(input_block_ptr, boundary_check=(1,))
    condition = tl.arange(0, block_size) < size
    input = tl.where(condition, input, float('-inf'))
    output = tl.argmax(input, 1)
    tl.store(output_block_ptr, output)
