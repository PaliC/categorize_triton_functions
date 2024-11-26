import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK': 1024}, num_stages=
    FORWARD_NUM_STAGES, num_warps=1), triton.Config({'BLOCK': 2048},
    num_stages=FORWARD_NUM_STAGES, num_warps=8), triton.Config({'BLOCK': 
    4096}, num_stages=FORWARD_NUM_STAGES, num_warps=8), triton.Config({
    'BLOCK': 8192}, num_stages=FORWARD_NUM_STAGES, num_warps=16), triton.
    Config({'BLOCK': 16384}, num_stages=FORWARD_NUM_STAGES, num_warps=16)],
    key=['N', 'CLASS_INDICES', 'log_size_logits', 'BUFFER_DTYPE'])
@triton.jit
def _forward(LOGITS, PROBS, IDX, LOSS, weight, N, WEIGHT_BUFFER,
    smoothing_factor, log_size_logits, WEIGHTS: 'tl.constexpr',
    CLASS_INDICES: 'tl.constexpr', LABEL_SMOOTHING: 'tl.constexpr',
    IGNORE_INDEX: 'tl.constexpr', BUFFER_DTYPE: 'tl.constexpr', BLOCK:
    'tl.constexpr'):
    if CLASS_INDICES:
        _class_indices_forward(LOGITS, PROBS, IDX, LOSS, weight, N,
            WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS,
            CLASS_INDICES, LABEL_SMOOTHING, IGNORE_INDEX, BUFFER_DTYPE, BLOCK)
    else:
        _class_probs_forward(LOGITS, PROBS, IDX, LOSS, weight, N,
            WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS,
            CLASS_INDICES, LABEL_SMOOTHING, IGNORE_INDEX, BUFFER_DTYPE, BLOCK)
