import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64, 'V_TILES': 1}, num_warps=8)],
    key=['V', 'N', 'H'], reset_to_zero=['losses_ptr', 'sumexp_ptr', 'z_nv_ptr']
    )
@triton.jit
def linear_xent_fwd_prep_bwd_kernel_matmul_t(x_ptr, y_ptr, A_t_ptr,
    z_nv_ptr, losses_ptr, sumexp_ptr, stride_x_N, stride_x_H, stride_A_H,
    stride_A_V, stride_z_N, stride_z_V, idx_N_group, N_group:
    'tl.constexpr', V: 'tl.constexpr', N: 'tl.constexpr', H: 'tl.constexpr',
    V_BLOCK_SIZE: 'tl.constexpr', N_BLOCK_SIZE: 'tl.constexpr',
    H_BLOCK_SIZE: 'tl.constexpr', V_TILES: 'tl.constexpr'=4):
    idx_N = tl.program_id(axis=0)
    idx_V_group = tl.program_id(axis=1)
    V_GROUP_SIZE: 'tl.constexpr' = V_TILES * V_BLOCK_SIZE
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(N, H), strides=(
        stride_x_N, stride_x_H), offsets=(idx_N_group * N_group + idx_N *
        N_BLOCK_SIZE, 0), block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE), order=(
        1, 0))
    A_block_ptr = tl.make_block_ptr(base=A_t_ptr, shape=(H, V), strides=(
        stride_A_H, stride_A_V), offsets=(0, idx_V_group * V_GROUP_SIZE),
        block_shape=(H_BLOCK_SIZE, V_BLOCK_SIZE), order=(1, 0))
    z_block_ptr = tl.make_block_ptr(base=z_nv_ptr, shape=(N_group, V),
        strides=(stride_z_N, stride_z_V), offsets=(idx_N * N_BLOCK_SIZE, 
        idx_V_group * V_GROUP_SIZE), block_shape=(N_BLOCK_SIZE,
        V_BLOCK_SIZE), order=(1, 0))
    sumexp_row_ptr = sumexp_ptr + idx_N * N_BLOCK_SIZE + tl.arange(0,
        N_BLOCK_SIZE)
    N_range = idx_N_group * N_group + idx_N * N_BLOCK_SIZE + tl.arange(0,
        N_BLOCK_SIZE)
    V_range = idx_V_group * V_GROUP_SIZE + tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + N_range)
    m = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32) - float(10000000.0)
    s = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32)
    loss = 0.0
    for _ in range(V_TILES):
        z_j_to_k = tl.zeros((N_BLOCK_SIZE, V_BLOCK_SIZE), dtype=tl.float32)
        for _ in range(H // H_BLOCK_SIZE):
            x_chunk = tl.load(x_block_ptr)
            A_v = tl.load(A_block_ptr)
            z_j_to_k = tl.dot(x_chunk, A_v, z_j_to_k)
            x_block_ptr = tl.advance(x_block_ptr, [0, H_BLOCK_SIZE])
            A_block_ptr = tl.advance(A_block_ptr, [H_BLOCK_SIZE, 0])
        m_new = tl.maximum(m, tl.max(z_j_to_k, 1))
        s_update = tl.sum(tl.exp(z_j_to_k - m_new[:, None]), axis=1)
        s = s * tl.exp(m - m_new) + s_update
        mask = y[:, None] == V_range[None, :]
        loss -= tl.sum(tl.where(mask, z_j_to_k, float(0.0))) / N
        tl.store(z_block_ptr, z_j_to_k)
        m = m_new
        x_block_ptr = tl.advance(x_block_ptr, [0, -H])
        A_block_ptr = tl.advance(A_block_ptr, [-H, V_BLOCK_SIZE])
        z_block_ptr = tl.advance(z_block_ptr, [0, V_BLOCK_SIZE])
        V_range = V_range + V_BLOCK_SIZE
    lse = m + tl.log(s)
    sum_exp = tl.exp(lse)
    tl.atomic_add(losses_ptr + idx_N, loss)
    tl.atomic_add(sumexp_row_ptr, sum_exp)
