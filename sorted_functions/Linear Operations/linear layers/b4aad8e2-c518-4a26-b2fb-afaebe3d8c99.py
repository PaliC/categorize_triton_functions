import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 128}, num_warps=8)], key=['V', 'N'], restore_value=[
    'z_nv_ptr'])
@triton.jit
def linear_xent_mini_bwd_prologue_kernel(z_nv_ptr, y_ptr, sumexp_ptr,
    stride_z_N, stride_z_V, idx_N_group, N_group: 'tl.constexpr', V:
    'tl.constexpr', N: 'tl.constexpr', V_BLOCK_SIZE: 'tl.constexpr',
    N_BLOCK_SIZE: 'tl.constexpr'):
    idx_N = tl.program_id(axis=0)
    idx_V = tl.program_id(axis=1)
    z_block_ptr = tl.make_block_ptr(base=z_nv_ptr, shape=(N_group, V),
        strides=(stride_z_N, stride_z_V), offsets=(idx_N * N_BLOCK_SIZE, 
        idx_V * V_BLOCK_SIZE), block_shape=(N_BLOCK_SIZE, V_BLOCK_SIZE),
        order=(1, 0))
    N_range = idx_N * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    v_range = idx_V * V_BLOCK_SIZE + tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + idx_N_group * N_group + N_range)
    lse = tl.log(tl.load(sumexp_ptr + N_range))
    z_j_to_k = tl.load(z_block_ptr)
    mask = y[:, None] == v_range[None, :]
    softmax_z = (z_j_to_k - lse[:, None]).exp()
    z_grad = (softmax_z - tl.where(mask, 1.0, 0.0)) / N
    tl.store(z_block_ptr, z_grad)
