import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_OUT': 256,
    'BLOCK_SIZE_HIDDEN': 64}, num_stages=3, num_warps=8), triton.Config({
    'BLOCK_SIZE_OUT': 256, 'BLOCK_SIZE_HIDDEN': 32}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_OUT': 128, 'BLOCK_SIZE_HIDDEN':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_OUT': 64,
    'BLOCK_SIZE_HIDDEN': 32}, num_stages=4, num_warps=4), triton.Config({
    'BLOCK_SIZE_OUT': 128, 'BLOCK_SIZE_HIDDEN': 32}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_OUT': 32, 'BLOCK_SIZE_HIDDEN':
    32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_OUT': 32,
    'BLOCK_SIZE_HIDDEN': 32}, num_stages=5, num_warps=2), triton.Config({
    'BLOCK_SIZE_OUT': 64, 'BLOCK_SIZE_HIDDEN': 32}, num_stages=5, num_warps
    =2)], key=['hidden_size', 'out_size'], restore_value=['weights_ptr'])
@triton.jit
def modifier_kernel(weights_ptr, assumed_wmax_ptr, reduced_assumed_wmax_ptr,
    upper_end_of_slices_ptr, hidden_size, out_size, num_slices,
    stride_weights_hidden_size, stride_weights_out_size,
    stride_assumed_wmax_num_slices, stride_assumed_wmax_out_size,
    modifier_type: 'tl.constexpr', modifier_weight_res: 'tl.constexpr',
    modifier_seed, modifier_std: 'tl.constexpr', BLOCK_SIZE_HIDDEN:
    'tl.constexpr', BLOCK_SIZE_OUT: 'tl.constexpr'):
    """
    Modifier kernel for the weights.
    """
    pid = tl.program_id(axis=0)
    offs_bn = (pid * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)) % out_size
    offs_assumed_wmax = pid * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    increase_weight_offsets_by = BLOCK_SIZE_HIDDEN * BLOCK_SIZE_OUT
    weight_random_offsets = tl.arange(0, BLOCK_SIZE_HIDDEN * BLOCK_SIZE_OUT
        ).reshape((BLOCK_SIZE_HIDDEN, BLOCK_SIZE_OUT), can_reorder=True)
    ir_range_lower = 0
    for slice_idx in range(0, num_slices):
        abs_max_slice_ptrs = (assumed_wmax_ptr + slice_idx *
            stride_assumed_wmax_num_slices + offs_bn *
            stride_assumed_wmax_out_size)
        if modifier_type == 'AddNormal' or (modifier_type == 'Discretize' or
            modifier_type == 'DiscretizeAddNormal'):
            assumed_wmax_per_slice = tl.load(reduced_assumed_wmax_ptr +
                slice_idx)
        else:
            assumed_wmax_per_slice = tl.load(abs_max_slice_ptrs, mask=
                offs_assumed_wmax < out_size, other=float('-inf'))
            assumed_wmax_per_slice = assumed_wmax_per_slice[None, :]
        ir_range_upper = tl.load(upper_end_of_slices_ptr + slice_idx)
        current_lower = ir_range_lower
        num_k = tl.cdiv(ir_range_upper - ir_range_lower, BLOCK_SIZE_HIDDEN)
        for k in range(0, num_k):
            current_upper = min(ir_range_upper, ir_range_lower + (k + 1) *
                BLOCK_SIZE_HIDDEN, hidden_size)
            offs_k = current_lower + tl.arange(0, BLOCK_SIZE_HIDDEN)
            b_ptrs = weights_ptr + (offs_k[:, None] *
                stride_weights_hidden_size + offs_bn[None, :] *
                stride_weights_out_size)
            weight_block = tl.load(b_ptrs, mask=offs_k[:, None] <
                current_upper, other=0.0)
            if (modifier_type == 'Discretize' or modifier_type ==
                'DiscretizeAddNormal') or (modifier_type ==
                'DiscretizePerChannel' or modifier_type ==
                'DiscretizeAddNormalPerChannel'):
                if modifier_weight_res > 0:
                    n_states = max(modifier_weight_res, 1 / modifier_weight_res
                        )
                    res = 2 * assumed_wmax_per_slice / n_states
                    weight_block = weight_block / res
                    weight_block = tl.extra.cuda.libdevice.rint(weight_block)
                    weight_block = weight_block * res
            if (modifier_type == 'AddNormal' or modifier_type ==
                'AddNormalPerChannel') or (modifier_type ==
                'DiscretizeAddNormal' or modifier_type ==
                'DiscretizeAddNormalPerChannel'):
                randn_block = tl.randn(modifier_seed + pid,
                    weight_random_offsets)
                weight_random_offsets += increase_weight_offsets_by
                randn_block = (assumed_wmax_per_slice * modifier_std *
                    randn_block)
                weight_block += randn_block
            tl.store(b_ptrs, weight_block, mask=(offs_k[:, None] <
                current_upper) & (offs_assumed_wmax[None, :] < out_size))
            current_lower = current_upper
        ir_range_lower = ir_range_upper
