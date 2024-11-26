import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_kernel(logits, lse, target, loss, total, ignore_index,
    label_smoothing: 'tl.constexpr', logit_scale: 'tl.constexpr', reduction:
    'tl.constexpr', V: 'tl.constexpr', BV: 'tl.constexpr'):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now.
    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Args:
        logits:
            Pointer to logits tensor.
        lse:
            Pointer to logsumexp tensor.
        target: Pointer to target tensor.
        loss:
            Pointer to tensor to store the loss.
        V (int):
            The number of columns in the input tensor.
        total (int):
            The number of non-ignored classes.
        ignore_index (int):
            The index to ignore in the target.
        label_smoothing (float):
            The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str):
            The string for the reduction to apply
        BV (int):
            The block size for vocab.
    """
    i_n = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    b_y = tl.load(target + i_n)
    logits += i_n * V
    if b_y == ignore_index:
        for i in range(0, V, BV):
            o_v = i + tl.arange(0, BV)
            tl.store(logits + o_v, 0.0, mask=o_v < V)
        return
    b_l = tl.load(logits + b_y) * logit_scale
    b_lse = tl.load(lse + i_n)
    b_loss = b_lse - b_l
    b_z = 0.0
    eps = label_smoothing / V
    tl.debug_barrier()
    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        b_logits = tl.load(logits + o_v, mask=o_v < V, other=float('-inf')
            ) * logit_scale
        if label_smoothing > 0:
            b_z += tl.sum(tl.where(o_v < V, -eps * b_logits, 0.0))
        b_p = (tl.exp(b_logits - b_lse) - eps) * logit_scale
        if reduction == 'mean':
            b_p = b_p / total
        tl.store(logits + o_v, b_p, mask=o_v < V)
        tl.debug_barrier()
    if label_smoothing > 0:
        b_loss = b_loss * (1 - label_smoothing) + (b_z + label_smoothing *
            b_lse)
    b_l = tl.load(logits + b_y)
    if reduction == 'mean':
        b_loss = b_loss / total
        b_l += (label_smoothing - 1) / total * logit_scale
    else:
        b_l += (label_smoothing - 1) * logit_scale
    tl.store(loss + i_n, b_loss)
    tl.store(logits + b_y, b_l)
