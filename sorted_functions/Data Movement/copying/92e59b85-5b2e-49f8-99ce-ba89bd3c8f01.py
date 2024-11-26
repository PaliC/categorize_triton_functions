import triton
import triton.language as tl
import torch

@eval(
    """triton.heuristics({
    'BLOCK_M': lambda kwargs: min(32, triton.next_power_of_2(kwargs['size_inp_0'])),
    'BLOCK_N': lambda kwargs: min(32, triton.next_power_of_2(kwargs['size_inp_1'])),
    'BLOCK_K': lambda kwargs: min(32, triton.next_power_of_2(kwargs['size_inp_2'])),
    'BATCH_STRIDE_INP_IS_1': lambda kwargs: kwargs['batch_stride_inp'] == 1,
    'STRIDE_INP_0_IS_1': lambda kwargs: kwargs['stride_inp_0'] == 1,
    'STRIDE_INP_1_IS_1': lambda kwargs: kwargs['stride_inp_1'] == 1,
    'STRIDE_INP_2_IS_1': lambda kwargs: kwargs['stride_inp_2'] == 1,
    'BATCH_STRIDE_OUT_IS_1': lambda kwargs: kwargs['batch_stride_out'] == 1,
    'STRIDE_OUT_0_IS_1': lambda kwargs: kwargs['stride_out_0'] == 1,
    'STRIDE_OUT_1_IS_1': lambda kwargs: kwargs['stride_out_1'] == 1,
    'STRIDE_OUT_2_IS_1': lambda kwargs: kwargs['stride_out_2'] == 1,
})"""
    )
@eval(
    """triton.heuristics({
    'num_warps': lambda kwargs: max(1, min(16, kwargs['BLOCK_M'] * kwargs['BLOCK_N'] * kwargs['BLOCK_K'] // 32)),
})"""
    )
@triton.jit
def copy_4d_kernel(output_ptr, input_ptr, bs, size_inp_0, size_inp_1,
    size_inp_2, batch_stride_inp, stride_inp_0, stride_inp_1, stride_inp_2,
    batch_stride_out, stride_out_0, stride_out_1, stride_out_2,
    BATCH_STRIDE_INP_IS_1: 'tl.constexpr', STRIDE_INP_0_IS_1:
    'tl.constexpr', STRIDE_INP_1_IS_1: 'tl.constexpr', STRIDE_INP_2_IS_1:
    'tl.constexpr', BATCH_STRIDE_OUT_IS_1: 'tl.constexpr',
    STRIDE_OUT_0_IS_1: 'tl.constexpr', STRIDE_OUT_1_IS_1: 'tl.constexpr',
    STRIDE_OUT_2_IS_1: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', BLOCK_K: 'tl.constexpr'):
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    grid_m = tl.cdiv(size_inp_0, BLOCK_M)
    grid_n = tl.cdiv(size_inp_1, BLOCK_N)
    grid_k = tl.cdiv(size_inp_2, BLOCK_K)
    pid_m = pid // (grid_n * grid_k)
    pid_nk = pid - pid_m * (grid_n * grid_k)
    pid_n = pid_nk // grid_k
    pid_k = pid_nk - pid_n * grid_k
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    A = input_ptr + (1 if BATCH_STRIDE_INP_IS_1 else batch_stride_inp
        ) * pid_batch + (rm[:, None, None] * (1 if STRIDE_INP_0_IS_1 else
        stride_inp_0) + rn[None, :, None] * (1 if STRIDE_INP_1_IS_1 else
        stride_inp_1) + rk[None, None, :] * (1 if STRIDE_INP_2_IS_1 else
        stride_inp_2))
    B = output_ptr + (1 if BATCH_STRIDE_OUT_IS_1 else batch_stride_out
        ) * pid_batch + (rm[:, None, None] * (1 if STRIDE_OUT_0_IS_1 else
        stride_out_0) + rn[None, :, None] * (1 if STRIDE_OUT_1_IS_1 else
        stride_out_1) + rk[None, None, :] * (1 if STRIDE_OUT_2_IS_1 else
        stride_out_2))
    mask = (rm < size_inp_0)[:, None, None] & (rn < size_inp_1)[None, :, None
        ] & (rk < size_inp_2)[None, None, :]
    a = tl.load(A, mask=mask)
    tl.store(B, a, mask=mask)
