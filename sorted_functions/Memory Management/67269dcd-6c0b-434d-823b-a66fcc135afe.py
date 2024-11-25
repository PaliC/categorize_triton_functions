import triton
import triton.language as tl
import torch

@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=['numels'])
@triton.jit
def dequant_kernel_248(g_idx_ptr, scales_ptr, qweight_ptr, qzeros_ptr,
    out_ptr, numels, maxq: 'tl.constexpr', bits: 'tl.constexpr',
    outfeatures: 'tl.constexpr', num_groups: 'tl.constexpr', X_BLOCK:
    'tl.constexpr'):
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    row_idx = x_index // outfeatures
    col_idx = x_index % outfeatures
    elements_per_feature: 'tl.constexpr' = 32 // bits
    g_idx = tl.load(g_idx_ptr + row_idx, None, eviction_policy='evict_last')
    qweights = tl.load(qweight_ptr + (col_idx + outfeatures * (row_idx //
        elements_per_feature)), None)
    wf_weights = row_idx % elements_per_feature * bits
    wf_zeros = col_idx % elements_per_feature * bits
    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    tl.device_assert(g_idx >= 0, 'index out of bounds: 0 <= tmp0 < 0')
    groups = tl.where(tmp2, tmp1, g_idx)
    scales = tl.load(scales_ptr + (col_idx + outfeatures * groups), None)
    weights = qweights >> wf_weights
    weights = weights & maxq
    qzero_ncols: 'tl.constexpr' = outfeatures // elements_per_feature
    qzeros = tl.load(qzeros_ptr + (qzero_ncols * groups + col_idx //
        elements_per_feature), None, eviction_policy='evict_last')
    zeros = qzeros >> wf_zeros
    zeros = zeros & maxq
    weights = weights - zeros
    weights = weights
    weights = scales * weights
    tl.store(out_ptr + x_index, weights, mask=xmask)
