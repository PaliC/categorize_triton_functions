import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64}), triton.Config({'V_BLOCK_SIZE':
    64, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 64}), triton.Config({
    'V_BLOCK_SIZE': 64, 'N_BLOCK_SIZE': 64, 'H_BLOCK_SIZE': 64}), triton.
    Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 256}),
    triton.Config({'V_BLOCK_SIZE': 512, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE':
    512}), triton.Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 64,
    'H_BLOCK_SIZE': 64}), triton.Config({'V_BLOCK_SIZE': 256,
    'N_BLOCK_SIZE': 256, 'H_BLOCK_SIZE': 64}), triton.Config({
    'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 256, 'H_BLOCK_SIZE': 256}), triton
    .Config({'V_BLOCK_SIZE': 256, 'N_BLOCK_SIZE': 16, 'H_BLOCK_SIZE': 16})],
    key=['V', 'N', 'H'], reset_to_zero=['losses_ptr', 'lse_ptr'])
@triton.jit
def linear_xent_fwd_kernel_matmul_t(x_ptr, y_ptr, A_t_ptr, losses_ptr,
    lse_ptr, stride_x_N, stride_x_H, stride_A_H, stride_A_V, V:
    'tl.constexpr', N: 'tl.constexpr', H: 'tl.constexpr', V_BLOCK_SIZE:
    'tl.constexpr', N_BLOCK_SIZE: 'tl.constexpr', H_BLOCK_SIZE: 'tl.constexpr'
    ):
    idx = tl.program_id(axis=0)
    tl.static_assert(N % N_BLOCK_SIZE == 0)
    tl.static_assert(V % V_BLOCK_SIZE == 0)
    tl.static_assert(H % H_BLOCK_SIZE == 0)
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(N, H), strides=(
        stride_x_N, stride_x_H), offsets=(idx * N_BLOCK_SIZE, 0),
        block_shape=(N_BLOCK_SIZE, H_BLOCK_SIZE), order=(1, 0))
    A_block_ptr = tl.make_block_ptr(base=A_t_ptr, shape=(H, V), strides=(
        stride_A_H, stride_A_V), offsets=(0, 0), block_shape=(H_BLOCK_SIZE,
        V_BLOCK_SIZE), order=(1, 0))
    offsets = idx * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)
    v_range = tl.arange(0, V_BLOCK_SIZE)
    y = tl.load(y_ptr + offsets)
    m = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32) - float(1000000.0)
    s = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32)
    loss = 0.0
    for _ in range(V // V_BLOCK_SIZE):
        z_j_to_k = tl.zeros((N_BLOCK_SIZE, V_BLOCK_SIZE), dtype=tl.float32)
        local_x_block_ptr = x_block_ptr
        for _ in range(H // H_BLOCK_SIZE):
            x_chunk = tl.load(local_x_block_ptr)
            A_v = tl.load(A_block_ptr)
            z_j_to_k = tl.dot(x_chunk, A_v, z_j_to_k)
            local_x_block_ptr = tl.advance(local_x_block_ptr, [0, H_BLOCK_SIZE]
                )
            A_block_ptr = tl.advance(A_block_ptr, [H_BLOCK_SIZE, 0])
        m_new = tl.maximum(m, tl.max(z_j_to_k, 1))
        s_update = tl.sum(tl.exp(z_j_to_k - m_new[:, None]), axis=1)
        s = s * tl.exp(m - m_new) + s_update
        mask = y[:, None] == v_range[None, :]
        loss -= tl.sum(tl.where(mask, z_j_to_k, float(0.0))) / N
        m = m_new
        A_block_ptr = tl.advance(A_block_ptr, [-H_BLOCK_SIZE * (H //
            H_BLOCK_SIZE), V_BLOCK_SIZE])
        v_range = v_range + V_BLOCK_SIZE
    lse = m + tl.log(s)
    loss += tl.sum(lse) / N
    tl.store(losses_ptr + idx, loss)
    tl.store(lse_ptr + offsets, lse)
