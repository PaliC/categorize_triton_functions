import triton
import triton.language as tl
import torch

@triton.jit
def gated_matmul_bwd_weights(input, y1_grad, y2_grad, dw1, dw2, M, N, K,
    stride_dom, stride_im, stride_wn, dtype: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', GROUP_N: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    BLOCK_K: 'tl.constexpr', IS_EVEN_MNK: 'tl.constexpr'):
    """
    Kernel for backward gated MLP
    We group along the M axis

    Ref :
    w1_grad = torch.matmul(y1_grad.t(), x)
    w2_grad = torch.matmul(y2_grad.t(), x)
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_N * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_N
    GROUP_N = min(num_pid_n - first_pid_n, GROUP_N)
    pid_n = first_pid_n + pid % GROUP_N
    pid_k = pid % num_pid_in_group // GROUP_N
    y1_grad_block_ptr = tl.make_block_ptr(base=y1_grad, shape=(N, M),
        strides=(1, stride_dom), offsets=(pid_n * BLOCK_N, 0), block_shape=
        (BLOCK_N, BLOCK_M), order=(0, 1))
    y2_grad_block_ptr = tl.make_block_ptr(base=y2_grad, shape=(N, M),
        strides=(1, stride_dom), offsets=(pid_n * BLOCK_N, 0), block_shape=
        (BLOCK_N, BLOCK_M), order=(0, 1))
    input_block_ptr = tl.make_block_ptr(base=input, shape=(M, K), strides=(
        stride_im, 1), offsets=(0, pid_k * BLOCK_K), block_shape=(BLOCK_M,
        BLOCK_K), order=(1, 0))
    ref = tl.load(input + tl.arange(0, 1))
    acc_dw1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc_dw2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for i in range(0, M, BLOCK_M):
        if IS_EVEN_MNK:
            y1grad_blk = tl.load(y1_grad_block_ptr)
            y2grad_blk = tl.load(y2_grad_block_ptr)
            x = tl.load(input_block_ptr)
        else:
            y1grad_blk = tl.load(y1_grad_block_ptr, boundary_check=(0, 1))
            y2grad_blk = tl.load(y2_grad_block_ptr, boundary_check=(0, 1))
            x = tl.load(input_block_ptr, boundary_check=(0, 1))
        acc_dw1 += tl.dot(y1grad_blk, x)
        acc_dw2 += tl.dot(y2grad_blk, x)
        y1_grad_block_ptr = tl.advance(y1_grad_block_ptr, (0, BLOCK_M))
        y2_grad_block_ptr = tl.advance(y2_grad_block_ptr, (0, BLOCK_M))
        input_block_ptr = tl.advance(input_block_ptr, (BLOCK_M, 0))
    dw1_ptrs = tl.make_block_ptr(base=dw1, shape=(N, K), strides=(stride_wn,
        1), offsets=(pid_n * BLOCK_N, pid_k * BLOCK_K), block_shape=(
        BLOCK_N, BLOCK_K), order=(1, 0))
    dw2_ptrs = tl.make_block_ptr(base=dw2, shape=(N, K), strides=(stride_wn,
        1), offsets=(pid_n * BLOCK_N, pid_k * BLOCK_K), block_shape=(
        BLOCK_N, BLOCK_K), order=(1, 0))
    if IS_EVEN_MNK:
        tl.store(dw1_ptrs, acc_dw1)
        tl.store(dw2_ptrs, acc_dw2)
    else:
        tl.store(dw1_ptrs, acc_dw1, boundary_check=(0, 1))
        tl.store(dw2_ptrs, acc_dw2, boundary_check=(0, 1))
