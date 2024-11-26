import triton
import triton.language as tl
import torch

@triton.jit
def _class_probs_forward(LOGITS, PROBS, IDX, LOSS, weight, N, WEIGHT_BUFFER,
    smoothing_factor, log_size_logits, WEIGHTS: 'tl.constexpr',
    CLASS_INDICES: 'tl.constexpr', LABEL_SMOOTHING: 'tl.constexpr',
    IGNORE_INDEX: 'tl.constexpr', BUFFER_DTYPE: 'tl.constexpr', BLOCK:
    'tl.constexpr'):
    buffer_dtype = _DTYPE2TRITON[BUFFER_DTYPE.value]
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    logit_start_ptrs = LOGITS + row * N
    logit_ptrs = logit_start_ptrs + cols
    m_prev = -float('inf')
    l_prev = 0.0
    m_prev = m_prev
    l_prev = l_prev
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(logit_ptrs, mask=cols < N - start_n * BLOCK,
            other=-float('inf'))
        m_curr = tl.maximum(tl.max(row_logits, 0), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(row_logits - m_curr)
        l_curr = tl.sum(p, 0) + l_prev
        l_prev = l_curr
        m_prev = m_curr
        logit_ptrs += BLOCK
    logit_ptrs = logit_start_ptrs + cols
    output_ptrs = PROBS + row * N + cols
    WRIT_PROBS = PROBS + row * N + cols
    sum_total = 0.0
    weights_total = 0.0
    sum_total = sum_total
    weights_total = weights_total
    idx_ptr = IDX + row * N + cols
    if WEIGHTS:
        weight_ptr = weight + cols
    l_prev_log = tl.log(l_prev)
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(logit_ptrs, mask=cols < N - start_n * BLOCK,
            other=l_prev_log + m_prev)
        idx = tl.load(idx_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
        full_weights_val = (1.0 - smoothing_factor
            ) * idx + smoothing_factor / N
        if WEIGHTS:
            weights_val = tl.load(weight_ptr, mask=cols < N - start_n *
                BLOCK, other=0.0)
            full_weights_val = weights_val * full_weights_val
        else:
            full_weights_val = tl.where(cols < N - start_n * BLOCK,
                full_weights_val, 0.0)
        weights_total += tl.sum(full_weights_val, 0)
        row_minus_max = row_logits - m_prev
        log_softmax = l_prev_log - row_minus_max
        log_softmax *= full_weights_val
        sum_total += tl.sum(log_softmax, 0)
        tl.store(WRIT_PROBS, log_softmax, mask=cols < N - start_n * BLOCK)
        logit_ptrs += BLOCK
        WRIT_PROBS += BLOCK
        idx_ptr += BLOCK
        if WEIGHTS:
            weight_ptr += BLOCK
    tl.store(WEIGHT_BUFFER + row, weights_total)
    probs = sum_total
    tl.store(LOSS + row, probs)
