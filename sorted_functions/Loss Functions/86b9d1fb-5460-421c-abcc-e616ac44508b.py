import triton
import triton.language as tl
import torch

@triton.heuristics({'DO_SOFTCAPPING': lambda args: args['DO_SOFTCAPPING'],
    'DO_LOGIT_SCALING': lambda args: args['DO_LOGIT_SCALING']})
@triton.jit
def _cross_entropy_backward(logits_ptr, logits_row_stride, dloss_ptr,
    dloss_row_stride, logsumexp_ptr, labels_ptr, VOCAB_SIZE: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr', DO_SOFTCAPPING: 'tl.constexpr', SOFTCAP:
    'tl.constexpr', DO_LOGIT_SCALING: 'tl.constexpr', LOGIT_SCALE:
    'tl.constexpr'):
    """
        CE_i = -y log(P) = y * (log[sum(exp(x))] - x)
        dC/dx = d/dx (y * log[sum(exp(x))] - x * y)

        From https://en.wikipedia.org/wiki/LogSumExp
        d/dx logsumexp = exp(x) / sum(exp(x)) = softmax(x)

        dC/dx = y * exp(x) / sum(exp(x)) - d/dx (x * y)
        dC/dx = y * exp[ log[exp(x) / sum(exp(x))] ] using x = exp(log(x)) trick
        dC/dx = y * exp[x - logsumexp] - d/dx (x * y)

        If y == 0: dC/dx = 0
        If y == 1 and x == label: dC/dlabel = exp[x - logsumexp] - 1
        If y == 1 and x != label: dC/dx     = exp[x - logsumexp]
    """
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0
    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf'))
    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE
    pass
    if DO_SOFTCAPPING:
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial
    pass
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x - logsumexp)
    y = tl.where(col_offsets == label_idx, y - 1.0, y)
    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE
    pass
    if DO_SOFTCAPPING:
        y = y * (1.0 - partial * partial)
    pass
    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)
