import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_INP': 128,
    'BLOCK_SIZE_OUT': 256, 'BLOCK_SIZE_HIDDEN': 64, 'GROUP_SIZE_INP': 8},
    num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_INP': 64,
    'BLOCK_SIZE_OUT': 256, 'BLOCK_SIZE_HIDDEN': 32, 'GROUP_SIZE_INP': 8},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_INP': 128,
    'BLOCK_SIZE_OUT': 128, 'BLOCK_SIZE_HIDDEN': 32, 'GROUP_SIZE_INP': 8},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_INP': 128,
    'BLOCK_SIZE_OUT': 64, 'BLOCK_SIZE_HIDDEN': 32, 'GROUP_SIZE_INP': 8},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_INP': 64,
    'BLOCK_SIZE_OUT': 128, 'BLOCK_SIZE_HIDDEN': 32, 'GROUP_SIZE_INP': 8},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_INP': 128,
    'BLOCK_SIZE_OUT': 32, 'BLOCK_SIZE_HIDDEN': 32, 'GROUP_SIZE_INP': 8},
    num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_INP': 64,
    'BLOCK_SIZE_OUT': 32, 'BLOCK_SIZE_HIDDEN': 32, 'GROUP_SIZE_INP': 8},
    num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_INP': 32,
    'BLOCK_SIZE_OUT': 64, 'BLOCK_SIZE_HIDDEN': 32, 'GROUP_SIZE_INP': 8},
    num_stages=5, num_warps=2)], key=['inp_size', 'hidden_size', 'out_size'])
@triton.jit
def matmul_kernel(inp_ptr, weights_ptr, out_ptr, ir_vector_ptr,
    assumed_wmax_ptr, reduced_assumed_wmax_ptr, input_range_ptr,
    upper_end_of_slices_ptr, inp_size, hidden_size, out_size, num_slices,
    stride_inp_inp_size, stride_inp_hidden_size, stride_weights_hidden_size,
    stride_weights_out_size, stride_out_inp_size, stride_out_out_size,
    stride_assumed_wmax_num_slices, stride_assumed_wmax_out_size,
    out_noise_seed, inp_res: 'tl.constexpr', is_fp: 'tl.constexpr', out_res:
    'tl.constexpr', out_quant: 'tl.constexpr', out_bound: 'tl.constexpr',
    bound_per_channel: 'tl.constexpr', out_noise: 'tl.constexpr',
    out_noise_std: 'tl.constexpr', out_noise_per_channel: 'tl.constexpr',
    ir_vector_is_none: 'tl.constexpr', dtype: 'tl.constexpr',
    BLOCK_SIZE_INP: 'tl.constexpr', BLOCK_SIZE_HIDDEN: 'tl.constexpr',
    BLOCK_SIZE_OUT: 'tl.constexpr', GROUP_SIZE_INP: 'tl.constexpr'):
    """
    Computes the block-wise matmul.
    Applies input range to the input and quantizes it. Converts
    back to the original range before accumulating the dot products.
    Can handle different input ranges per slice in the input dimension.
    Stores the MVM result inp_ptr @ weights_ptr in out_ptr.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(inp_size, BLOCK_SIZE_INP)
    num_pid_n = tl.cdiv(out_size, BLOCK_SIZE_OUT)
    num_pid_in_group = GROUP_SIZE_INP * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_INP
    GROUP_SIZE_INP = min(num_pid_m - first_pid_m, GROUP_SIZE_INP)
    pid_m = first_pid_m + pid % num_pid_in_group % GROUP_SIZE_INP
    pid_n = pid % num_pid_in_group // GROUP_SIZE_INP
    accumulator = tl.zeros((BLOCK_SIZE_INP, BLOCK_SIZE_OUT), dtype=tl.float32)
    per_slice_accumulator = tl.zeros((BLOCK_SIZE_INP, BLOCK_SIZE_OUT),
        dtype=tl.float32)
    increase_out_offsets_by = BLOCK_SIZE_INP * BLOCK_SIZE_OUT
    output_random_offsets = tl.arange(0, BLOCK_SIZE_INP * BLOCK_SIZE_OUT
        ).reshape((BLOCK_SIZE_INP, BLOCK_SIZE_OUT), can_reorder=True)
    offs_am = pid_m * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)
    offs_bn = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_INP),
        BLOCK_SIZE_INP)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_OUT),
        BLOCK_SIZE_OUT)
    offs_assumed_wmax = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    ir_range_lower = 0
    for slice_idx in range(0, num_slices):
        if slice_idx > 0:
            per_slice_accumulator = tl.zeros((BLOCK_SIZE_INP,
                BLOCK_SIZE_OUT), dtype=tl.float32)
        abs_max_slice_ptrs = (assumed_wmax_ptr + slice_idx *
            stride_assumed_wmax_num_slices + offs_bn *
            stride_assumed_wmax_out_size)
        if out_noise and out_noise_per_channel:
            assumed_wmax_per_slice = tl.load(abs_max_slice_ptrs, mask=
                offs_assumed_wmax < out_size, other=float('-inf'))
            assumed_wmax_per_slice = assumed_wmax_per_slice[None, :]
        else:
            assumed_wmax_per_slice = tl.load(reduced_assumed_wmax_ptr +
                slice_idx)
        if bound_per_channel and not (out_noise and out_noise_per_channel):
            bound_scale = tl.load(abs_max_slice_ptrs, mask=
                offs_assumed_wmax < out_size, other=float('-inf'))
            bound_scale = bound_scale[None, :]
        else:
            bound_scale = assumed_wmax_per_slice
        if num_slices == 1:
            ir_range_upper = hidden_size
        else:
            ir_range_upper = tl.load(upper_end_of_slices_ptr + slice_idx)
        current_lower = ir_range_lower
        input_range = tl.load(input_range_ptr + slice_idx)
        offs_k = current_lower + tl.arange(0, BLOCK_SIZE_HIDDEN)
        a_ptrs = inp_ptr + (offs_am[:, None] * stride_inp_inp_size + offs_k
            [None, :] * stride_inp_hidden_size)
        b_ptrs = weights_ptr + (offs_k[:, None] *
            stride_weights_hidden_size + offs_bn[None, :] *
            stride_weights_out_size)
        num_k = tl.cdiv(ir_range_upper - ir_range_lower, BLOCK_SIZE_HIDDEN)
        for k in range(0, num_k):
            current_upper = min(ir_range_upper, ir_range_lower + (k + 1) *
                BLOCK_SIZE_HIDDEN, hidden_size)
            inp_block = tl.load(a_ptrs, mask=(offs_am[:, None] < inp_size) &
                (offs_k[None, :] < current_upper), other=0.0)
            weight_block = tl.load(b_ptrs, mask=(offs_k[:, None] <
                current_upper) & (offs_bn[None, :] < out_size), other=0.0)
            input_range = input_range
            if not ir_vector_is_none:
                tl.store(ir_vector_ptr + offs_k, input_range, mask=offs_k <
                    current_upper)
            over_ir_mask = inp_block > input_range
            under_ir_mask = inp_block < -input_range
            inp_block = tl.where(over_ir_mask, input_range, inp_block)
            inp_block = tl.where(under_ir_mask, -input_range, inp_block)
            inp_block = inp_block / input_range
            if not is_fp:
                inp_block = inp_block / inp_res
                inp_block = tl.extra.cuda.libdevice.rint(inp_block)
                inp_block = inp_block * inp_res
            inp_block = inp_block
            dot_prod = tl.dot(inp_block, weight_block)
            dot_prod = input_range * dot_prod
            per_slice_accumulator = per_slice_accumulator + dot_prod
            offs_k += current_upper - current_lower
            a_ptrs += (current_upper - current_lower) * stride_inp_hidden_size
            b_ptrs += (current_upper - current_lower
                ) * stride_weights_hidden_size
            current_lower = current_upper
        if out_noise:
            randn_block = tl.randn(out_noise_seed + pid, output_random_offsets)
            randn_block = assumed_wmax_per_slice * out_noise_std / tl.sqrt(
                num_slices * num_k) * randn_block
            per_slice_accumulator += randn_block
            output_random_offsets += increase_out_offsets_by
        if out_quant or out_bound > 0:
            bound = bound_scale * out_bound * input_range
            if out_quant:
                alpha = bound * out_res
                per_slice_accumulator = per_slice_accumulator / tl.where(
                    alpha == 0.0, FLOAT32_TINY, alpha)
                per_slice_accumulator = tl.extra.cuda.libdevice.rint(
                    per_slice_accumulator)
                per_slice_accumulator = per_slice_accumulator * alpha
            if out_bound > 0:
                over_out_bound_mask = per_slice_accumulator > bound
                under_out_bound_mask = per_slice_accumulator < -bound
                per_slice_accumulator = tl.where(over_out_bound_mask, bound,
                    per_slice_accumulator)
                per_slice_accumulator = tl.where(under_out_bound_mask, -
                    bound, per_slice_accumulator)
        accumulator = accumulator + per_slice_accumulator
        ir_range_lower = ir_range_upper
    out_block = accumulator
    offs_cm = pid_m * BLOCK_SIZE_INP + tl.arange(0, BLOCK_SIZE_INP)
    offs_cn = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    c_ptrs = out_ptr + stride_out_inp_size * offs_cm[:, None
        ] + stride_out_out_size * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < inp_size) & (offs_cn[None, :] < out_size)
    tl.store(c_ptrs, out_block, mask=c_mask)
