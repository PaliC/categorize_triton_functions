import triton
import triton.language as tl
import torch

@triton.heuristics(values={'USE_MASK': lambda args: args['numels'] % args[
    'BLOCK_SIZE'] != 0, 'NUM_GROUPS': lambda args: triton.cdiv(args[
    'numels'], args['BLOCK_SIZE'])})
@triton.jit
def _quantize_blockwise_kernel(t_ptr, cutoffs_ptr, q_ptr, absmax_ptr,
    norm_ptr, numels, BLOCK_SIZE: 'tl.constexpr', NUM_BUCKETS:
    'tl.constexpr', USE_MASK: 'tl.constexpr', NUM_GROUPS: 'tl.constexpr',
    RETURN_NORM: 'tl.constexpr'=False):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = None
    absmax_mask = None
    if USE_MASK:
        mask = offsets < numels
        absmax_mask = pid < NUM_GROUPS
    t = tl.load(t_ptr + offsets, mask=mask)
    absmax = tl.max(tl.abs(t), axis=0)
    normalized = t / absmax
    cutoffs = tl.load(cutoffs_ptr + tl.arange(0, NUM_BUCKETS))
    q = tl.reshape(normalized, (BLOCK_SIZE, 1)) > cutoffs
    q = q
    q = tl.sum(q, axis=1)
    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(absmax_ptr + pid, absmax, mask=absmax_mask)
    if RETURN_NORM:
        tl.store(norm_ptr + offsets, normalized, mask=mask)
