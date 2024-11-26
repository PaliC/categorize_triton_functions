import triton
import triton.language as tl
import torch

@triton.jit
def gated_matmul_bwd_input(w1, w2, y1_grad, y2_grad, din, M, N, K,
    stride_dom, stride_im, stride_wn, dtype: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', GROUP_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    BLOCK_K: 'tl.constexpr', IS_EVEN_MNK: 'tl.constexpr'):
    """
    Kernel for backward gated MLP
    We group along the N axis

    Ref :
    x_grad = torch.matmul(y2_grad, w2.t()) + torch.matmul(y1_grad, w1.t())
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % GROUP_M
    pid_k = pid % num_pid_in_group // GROUP_M
    y1_grad_block_ptr = tl.make_block_ptr(base=y1_grad, shape=(M, N),
        strides=(stride_dom, 1), offsets=(pid_m * BLOCK_M, 0), block_shape=
        (BLOCK_M, BLOCK_N), order=(1, 0))
    y2_grad_block_ptr = tl.make_block_ptr(base=y2_grad, shape=(M, N),
        strides=(stride_dom, 1), offsets=(pid_m * BLOCK_M, 0), block_shape=
        (BLOCK_M, BLOCK_N), order=(1, 0))
    w1_block_ptr = tl.make_block_ptr(base=w1, shape=(N, K), strides=(
        stride_wn, 1), offsets=(0, pid_k * BLOCK_K), block_shape=(BLOCK_N,
        BLOCK_K), order=(1, 0))
    w2_block_ptr = tl.make_block_ptr(base=w2, shape=(N, K), strides=(
        stride_wn, 1), offsets=(0, pid_k * BLOCK_K), block_shape=(BLOCK_N,
        BLOCK_K), order=(1, 0))
    acc_dx = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for i in range(0, N, BLOCK_N):
        if IS_EVEN_MNK:
            w1_blk = tl.load(w1_block_ptr)
            w2_blk = tl.load(w2_block_ptr)
            y1_grad_blk = tl.load(y1_grad_block_ptr)
            y2_grad_blk = tl.load(y2_grad_block_ptr)
        else:
            w1_blk = tl.load(w1_block_ptr, boundary_check=(0, 1))
            w2_blk = tl.load(w2_block_ptr, boundary_check=(0, 1))
            y1_grad_blk = tl.load(y1_grad_block_ptr, boundary_check=(0, 1))
            y2_grad_blk = tl.load(y2_grad_block_ptr, boundary_check=(0, 1))
        acc_dx += tl.dot(y2_grad_blk, w2_blk)
        acc_dx += tl.dot(y1_grad_blk, w1_blk)
        w1_block_ptr = tl.advance(w1_block_ptr, (BLOCK_N, 0))
        w2_block_ptr = tl.advance(w2_block_ptr, (BLOCK_N, 0))
        y1_grad_block_ptr = tl.advance(y1_grad_block_ptr, (0, BLOCK_N))
        y2_grad_block_ptr = tl.advance(y2_grad_block_ptr, (0, BLOCK_N))
    dx_ptrs = tl.make_block_ptr(base=din, shape=(M, K), strides=(stride_im,
        1), offsets=(pid_m * BLOCK_M, pid_k * BLOCK_K), block_shape=(
        BLOCK_M, BLOCK_K), order=(1, 0))
    if IS_EVEN_MNK:
        tl.store(dx_ptrs, acc_dx)
    else:
        tl.store(dx_ptrs, acc_dx, boundary_check=(0, 1))
