import triton
import triton.language as tl
import torch

@triton.jit
def _silu_and_mul_kernel(input_ptr, stride_input_m, stride_input_n,
    stride_output_m, stride_output_n, size_m, size_n, BLOCK_M:
    'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    stride_input_m = stride_input_m
    stride_output_m = stride_output_m
    tid = tl.program_id(0)
    input_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    output_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    pid = tl.program_id(1)
    input_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    output_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    up_offsets = input_m_offsets[:, None] * stride_input_m + (input_n_offsets
        [None, :] + size_n) * stride_input_n
    gate_offsets = input_m_offsets[:, None] * stride_input_m + input_n_offsets[
        None, :] * stride_input_n
    res_offsets = output_m_offsets[:, None
        ] * stride_output_m + output_n_offsets[None, :] * stride_output_n
    up = tl.load(input_ptr + up_offsets, mask=(input_n_offsets < size_n)[
        None, :] * (input_m_offsets < size_m)[:, None], other=0.0)
    gate = tl.load(input_ptr + gate_offsets, mask=(input_n_offsets < size_n
        )[None, :] * (input_m_offsets < size_m)[:, None], other=0.0)
    gate = gate / (1 + tl.exp(-gate))
    gate = gate
    tl.store(input_ptr + res_offsets, up * gate, mask=(output_n_offsets <
        size_n)[None, :] * (output_m_offsets < size_m)[:, None])
