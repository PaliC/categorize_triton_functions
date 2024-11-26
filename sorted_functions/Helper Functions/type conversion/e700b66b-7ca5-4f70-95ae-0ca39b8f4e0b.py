import triton
import triton.language as tl
import torch

@triton.jit
def liger_cross_entropy_kernel(X_ptr, X_stride, Y_ptr, Y_stride, loss_ptr,
    z_loss_ptr, loss_stride, n_cols, n_non_ignore, ignore_index,
    lse_square_scale: 'tl.constexpr', label_smoothing: 'tl.constexpr',
    reduction: 'tl.constexpr', softcap, RETURN_Z_LOSS: 'tl.constexpr',
    BLOCK_SIZE: 'tl.constexpr', HAS_SOFTCAPPING: 'tl.constexpr'):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    loss_ptr: Pointer to tensor to store the loss.
    z_loss_ptr: Pointer to tensor to store the z loss. No operation if RETURN_Z_LOSS is 0.
    loss_stride (int): The stride of the loss tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (int): The number of non-ignored elements in the batch.
    ignore_index (int): The index to ignore in the target.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
    RETURN_Z_LOSS (int): The boolean value to decide whether storing z loss to z_loss_ptr or not. It must be 0 or 1.
    reduction (str): The string for the reduction to apply
    softcap (float): The upper threshold for scaling logits to the range (-softcap, +softcap).
    BLOCK_SIZE (int): The block size for Triton operations.
    HAS_SOFTCAPPING (bool): The boolean value to determine whether applying soft-capping or not.
    """
    program_id = tl.program_id(0)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    X_ptr += program_id * X_stride
    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return
    loss_ptr += program_id * loss_stride
    z_loss_ptr += program_id * loss_stride
    m = float('-inf')
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)
    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tanh(ori_X_y / softcap)
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other
            =float('-inf')).cast(tl.float32)
        if HAS_SOFTCAPPING:
            X_block = softcap * tanh(X_block / softcap)
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps *
                X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new
    lse = m + tl.log(d)
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other
            =float('-inf')).cast(tl.float32)
        if HAS_SOFTCAPPING:
            intermediate = tanh(X_block / softcap)
            X_block = softcap * intermediate
        X_block = tl.exp(X_block - m) / d
        X_block += 2 * lse_square_scale * lse * X_block
        X_block += -eps
        X_block = tl.where(X_offsets != y, X_block, X_block - (1 -
            label_smoothing))
        if reduction == 'mean':
            X_block = X_block / n_non_ignore
        if HAS_SOFTCAPPING:
            X_block = X_block * (1 - intermediate * intermediate)
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)
    tl.debug_barrier()
    loss = lse - ori_X_y
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * lse
        loss = loss * (1 - label_smoothing) + smooth_loss
    z_loss = lse_square_scale * lse * lse
    loss += z_loss
    if reduction == 'mean':
        z_loss = z_loss / n_non_ignore
        loss = loss / n_non_ignore
    tl.store(loss_ptr, loss)
    if RETURN_Z_LOSS == _TRUE:
        tl.store(z_loss_ptr, z_loss)
