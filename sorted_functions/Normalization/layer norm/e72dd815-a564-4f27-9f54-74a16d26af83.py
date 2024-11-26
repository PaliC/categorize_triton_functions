import triton
import triton.language as tl
import torch

@triton.jit
def forward(output_ptr: 'tl.tensor', input_ptr: 'tl.tensor', rstd_ptr:
    'tl.tensor', mean_ptr: 'tl.tensor', group_size, y_size, x_size,
    num_groups, weight_ptr: 'tl.tensor', bias_ptr: 'tl.tensor', eps, dtype:
    'tl.constexpr', group_block_size: 'tl.constexpr', x_block_size:
    'tl.constexpr', ACTIVATION: 'tl.constexpr'):
    pid = tl.program_id(0)
    batch = pid // num_groups
    group = pid % num_groups
    num_elements = group_size * x_size
    batch_offset = batch * num_groups * num_elements
    group_offset = batch_offset + group * num_elements
    output_block_ptr = tl.make_block_ptr(output_ptr + group_offset, shape=(
        group_size, x_size), strides=(x_size, 1), offsets=(0, 0),
        block_shape=(group_block_size, x_block_size), order=(1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr + group_offset, shape=(
        group_size, x_size), strides=(x_size, 1), offsets=(0, 0),
        block_shape=(group_block_size, x_block_size), order=(1, 0))
    rstd_block_ptr = tl.make_block_ptr(rstd_ptr + batch * num_groups, shape
        =(group_size,), strides=(1,), offsets=(group,), block_shape=(1,),
        order=(0,))
    mean_block_ptr = tl.make_block_ptr(mean_ptr + batch * num_groups, shape
        =(group_size,), strides=(1,), offsets=(group,), block_shape=(1,),
        order=(0,))
    input = tl.load(input_block_ptr)
    mean = tl.sum(tl.view(input / num_elements, (1, group_block_size *
        x_block_size)), 1)
    centered_mean = input - mean
    var = tl.sum(tl.view(centered_mean * centered_mean / num_elements, (1, 
        group_block_size * x_block_size)), 1)
    rstd = tl.math.rsqrt(var + eps)
    output = centered_mean * rstd
    if weight_ptr is not None:
        weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(y_size, 1),
            strides=(1, y_size), offsets=(group * group_size, 0),
            block_shape=(group_block_size, 1), order=(0, 1))
        weight = tl.load(weight_block_ptr, boundary_check=(0,))
        output *= weight
    if bias_ptr is not None:
        bias_block_ptr = tl.make_block_ptr(bias_ptr, shape=(y_size, 1),
            strides=(1, y_size), offsets=(group * group_size, 0),
            block_shape=(group_block_size, 1), order=(0, 1))
        bias = tl.load(bias_block_ptr, boundary_check=(0,))
        output += bias
    if ACTIVATION:
        output = silu(output)
    tl.store(output_block_ptr, output)
    tl.store(rstd_block_ptr, rstd)
    tl.store(mean_block_ptr, mean)
