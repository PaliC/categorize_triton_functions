import triton
import triton.language as tl
import torch

@triton.jit
def _softmax_grad_core(output_ptrs, d_output_ptrs, d_input_ptrs,
    col_offsets, n_cols, is_bf16: 'tl.constexpr'):
    output_row = tl.load(output_ptrs, mask=col_offsets < n_cols, other=float(0)
        )
    d_output_row = tl.load(d_output_ptrs, mask=col_offsets < n_cols, other=
        float(0))
    if is_bf16:
        output_row = output_row
        d_output_row = d_output_row
    row_sum = tl.sum(output_row * d_output_row, axis=0)
    d_softmax_output = (d_output_row - row_sum) * output_row
    tl.store(d_input_ptrs, d_softmax_output, mask=col_offsets < n_cols)
