import triton
import triton.language as tl
import torch

@triton.heuristics({'HAS_DT_BIAS': lambda args: args['dt_bias_ptr'] is not
    None})
@triton.heuristics({'HAS_D': lambda args: args['D_ptr'] is not None})
@triton.heuristics({'HAS_Z': lambda args: args['z_ptr'] is not None})
@triton.heuristics({'BLOCK_SIZE_DSTATE': lambda args: triton.
    next_power_of_2(args['dstate'])})
@triton.jit
def _selective_scan_update_kernel(state_ptr, x_ptr, dt_ptr, dt_bias_ptr,
    A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr, batch, dim, dstate,
    stride_state_batch, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_dim, stride_dt_batch, stride_dt_dim,
    stride_dt_bias_dim, stride_A_dim, stride_A_dstate, stride_B_batch,
    stride_B_dstate, stride_C_batch, stride_C_dstate, stride_D_dim,
    stride_z_batch, stride_z_dim, stride_out_batch, stride_out_dim,
    DT_SOFTPLUS: 'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr', HAS_DT_BIAS:
    'tl.constexpr', HAS_D: 'tl.constexpr', HAS_Z: 'tl.constexpr',
    BLOCK_SIZE_DSTATE: 'tl.constexpr'):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    state_ptr += pid_b * stride_state_batch
    x_ptr += pid_b * stride_x_batch
    dt_ptr += pid_b * stride_dt_batch
    B_ptr += pid_b * stride_B_batch
    C_ptr += pid_b * stride_C_batch
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch
    out_ptr += pid_b * stride_out_batch
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[
        None, :] * stride_state_dstate)
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] *
        stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None,
        :] < dstate), other=0.0)
    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0)
    dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0)
    if HAS_DT_BIAS:
        dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0)
    if DT_SOFTPLUS:
        dt = tl.log(1.0 + tl.exp(dt))
    A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] <
        dstate), other=0.0)
    dA = tl.exp(A * dt[:, None])
    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0)
    dB = B[None, :] * dt[:, None]
    state = state * dA + dB * x[:, None]
    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None,
        :] < dstate))
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=offs_m < dim)
