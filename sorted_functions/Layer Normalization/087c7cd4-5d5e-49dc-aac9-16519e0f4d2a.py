import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, out_ptr, stride_x_batch, stride_x_m,
    stride_x_k, stride_rms_w, stride_out_batch, stride_out_m, stride_out_k,
    N_SIZE: 'tl.constexpr', eps: 'tl.constexpr', BLOCK_N_SIZE: 'tl.constexpr'):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    offset_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_n_size = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=
            x_ptr_mask, other=0.0)
        xf = x
        var += xf * xf
    var = tl.sum(var, axis=0) / N_SIZE
    std = tl.sqrt(var + eps)
    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE
        rms_w_offset = tl.load(rms_w_ptr + offset_n * stride_rms_w, mask=
            x_ptr_mask)
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=
            x_ptr_mask, other=0.0)
        x_new = x / std
        out = x_new * rms_w_offset
        out_offset = (pid_batch * stride_out_batch + pid_m * stride_out_m +
            offset_n * stride_out_k)
        tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)
