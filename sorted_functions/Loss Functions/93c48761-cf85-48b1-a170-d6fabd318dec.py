import triton
import triton.language as tl
import torch

@triton.heuristics({'DO_SOFTCAPPING': lambda args: args['DO_SOFTCAPPING'],
    'DO_LOGIT_SCALING': lambda args: args['DO_LOGIT_SCALING']})
@triton.jit
def _chunked_cross_entropy_forward(logits_ptr, logits_row_stride, loss_ptr,
    logsumexp_ptr, labels_ptr, VOCAB_SIZE: 'tl.constexpr', N_CHUNKS:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', DO_SOFTCAPPING:
    'tl.constexpr', SOFTCAP: 'tl.constexpr', DO_LOGIT_SCALING:
    'tl.constexpr', LOGIT_SCALE: 'tl.constexpr'):
    """
        256K vocab divided in 4 chunks

        |-65536-| |-65536-| |-65536-| |-65536-|
        |-------| |-------| |-------| |-------|
        |-------| |-------| |-------| |-------|

        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        Notice we can do logsumexp for each chunk and then
        logsumexp[chunk_sum(logsumexp)] == logsumexp

        chunk_sum = log[chunk_sum(logsumexp)]
                  = log[exp(logsumexp(a)) + ... + exp(logsumexp(z))]
                  = log[exp(log[sum(exp(a))]) + ... + exp(log[sum(exp(z))])]
                  = log[sum(exp(a)) + ... + sum(exp(z))]
                  = logsumexp(x)

        This means we can perform a logsumexp for each chunk, then do a
        final logsumexp reduction!

        Ie do: logsumexp(chunked_logsumexp) - x
    """
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx
    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf'))
    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)
    logits = logits
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx)
            if DO_LOGIT_SCALING:
                x = LOGIT_SCALE * x
            if DO_SOFTCAPPING:
                x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    pass
    tl.store(logsumexp_ptr, logsumexp)
