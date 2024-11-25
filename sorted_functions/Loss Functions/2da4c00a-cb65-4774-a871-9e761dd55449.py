import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK': 1024}, num_stages=1,
    num_warps=1), triton.Config({'BLOCK': 2048}, num_stages=1, num_warps=8),
    triton.Config({'BLOCK': 4096}, num_stages=1, num_warps=8), triton.
    Config({'BLOCK': 8192}, num_stages=1, num_warps=16), triton.Config({
    'BLOCK': 16384}, num_stages=1, num_warps=16)], key=['N',
    'CLASS_INDICES', 'log_size_logits', 'BUFFER_DTYPE'])
@triton.jit
def _backward(PROBS, IDX, DPROBS, dprob_stride, DIN, weight, N,
    WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS:
    'tl.constexpr', CLASS_INDICES: 'tl.constexpr', LABEL_SMOOTHING:
    'tl.constexpr', IGNORE_INDEX: 'tl.constexpr', BUFFER_DTYPE:
    'tl.constexpr', BLOCK: 'tl.constexpr'):
    buffer_dtype = _DTYPE2TRITON[BUFFER_DTYPE.value]
    row = tl.program_id(0)
    start_n = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    PROBS = PROBS + row * N
    probs_start = PROBS + cols + BLOCK * start_n
    probs = -tl.load(probs_start, mask=cols < N - start_n * BLOCK, other=
        float('inf'))
    DIN = DIN + row * N + cols + BLOCK * start_n
    dout = tl.load(DPROBS + row * dprob_stride)
    if CLASS_INDICES:
        idx = tl.load(IDX + row)
        delta = start_n * BLOCK + cols == idx
        if IGNORE_INDEX >= 0:
            use_class = idx == IGNORE_INDEX
            dout = dout * (1 - use_class)
        if LABEL_SMOOTHING:
            if WEIGHTS:
                weight_ptr = weight + cols + BLOCK * start_n
                full_weights_val = tl.load(weight_ptr, mask=cols < N - 
                    start_n * BLOCK, other=0.0)
                weights_val = tl.load(weight + idx)
                probs = probs / full_weights_val
            probs = tl.exp(probs)
            if WEIGHTS:
                weights_total = tl.load(WEIGHT_BUFFER + row)
                numerator_contrib = weights_val * (1.0 - smoothing_factor) * (
                    probs - delta)
                mean_contrib = (weights_total * probs - full_weights_val
                    ) * smoothing_factor / N
            else:
                numerator_contrib = (1.0 - smoothing_factor) * (probs - delta)
                mean_contrib = smoothing_factor * probs - smoothing_factor / N
            din = (numerator_contrib + mean_contrib) * dout
        else:
            probs = tl.exp(probs)
            din = (probs - delta) * dout
            if WEIGHTS:
                weight_ptr = weight + idx
                weights_val = tl.load(weight_ptr)
                din = weights_val * din
    else:
        idx = tl.load(IDX + row * N + cols + BLOCK * start_n, mask=cols < N -
            start_n * BLOCK, other=0.0)
        full_weights_val = (1.0 - smoothing_factor
            ) * idx + smoothing_factor / N
        weights_total = tl.load(WEIGHT_BUFFER + row)
        if WEIGHTS:
            weight_ptr = weight + cols + BLOCK * start_n
            weights_val = tl.load(weight_ptr, mask=cols < N - start_n *
                BLOCK, other=0.0)
            full_weights_val = weights_val * full_weights_val
        probs = probs / full_weights_val
        probs = tl.exp(probs)
        weighted_probs = probs * weights_total
        weighted_probs_per_class = weighted_probs - full_weights_val
        din = weighted_probs_per_class * dout
    tl.store(DIN, din, mask=cols + BLOCK * start_n < N)
