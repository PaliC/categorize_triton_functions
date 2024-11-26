import triton
import triton.language as tl
import torch

@triton.jit
def causal_product_bwd_kernel(q_ptr, k_ptr, v_ptr, grad_out, grad_Q_ptr,
    grad_K_ptr, grad_V_ptr, batch, length, dim, vdim, **meta):
    BLOCK_SIZE = meta['BLOCK_SIZE']
    pid = tl.program_id(axis=0)
    state = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    cur_qk_pos = pid * matrix_size * dim
    cur_v_pos = pid * matrix_size * vdim
    dim_ptrs = tl.arange(0, BLOCK_SIZE)
    qkmask = dim_ptrs < dim
    vmask = dim_ptrs < vdim
    for _ in range(0, length, 1):
        qk_row_offsets = cur_qk_pos + dim_ptrs
        v_row_offsets = cur_v_pos + dim_ptrs
        k = tl.load(k_ptr + qk_row_offsets, mask=qkmask, other=0)
        v = tl.load(v_ptr + v_row_offsets, mask=vmask, other=0)
        context = tl.dot(k[:, None], v[None, :])
        state += context
        g = tl.load(grad_out + v_row_offsets, mask=vmask, other=0)
        grad_q = tl.dot(state, g[:, None])
        tl.store(grad_Q_ptr + qk_row_offsets[:, None], grad_q, mask=qkmask[
            :, None])
        cur_qk_pos += dim
        cur_v_pos += vdim
    """
    state *= 0

    for _ in range(0, length, 1):
        # Move back one row
        cur_pos -= dim

        # Offset for a single row in Q, K, V
        row_offsets = cur_pos + dim_ptrs

        # Load the current row of Q, K, V vectors. All are vectors of shape [dim]
        q = tl.load(q_ptr + row_offsets, mask=mask, other=0)
        k = tl.load(k_ptr + row_offsets, mask=mask, other=0)
        v = tl.load(v_ptr + row_offsets, mask=vmask, other=0)
        # Load gradient
        g = tl.load(grad_out + row_offsets, mask=vmask, other=0)
        # Compute context [D, M] matrix from [D, 1] x [1, M]
        context = tl.dot(q[:, None], g[None, :])
        # state += context

        # Compute gradients [1, D] x [D, M] => [1, M]
        grad_v = tl.dot(k[None, :], context)
        grad_v = tl.reshape(grad_v, (meta['BLOCK_SIZE'],))
        # grad_v = tl.dot(k[None, :], state)

        # Enabling the follownig leads to a hang

        # grad_k = tl.dot(state, v[:, None])
        # print(grad_v.shape)
        # print(grad_k.shape)
        # Store the result of this row
        # tl.store(grad_V_ptr + row_offsets[None,
        #          :], grad_v, mask=vmask[None, :])
        tl.store(grad_V_ptr + row_offsets, grad_v, mask=vmask)
        # tl.store(grad_K_ptr + row_offsets[:, None], grad_k, mask=mask[:, None])
    """
