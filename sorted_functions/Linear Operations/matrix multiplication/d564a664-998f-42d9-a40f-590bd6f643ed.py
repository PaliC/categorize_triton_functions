import triton
import triton.language as tl
import torch

@triton.jit
def gated_matmul_fwd(out, input, w1, w2, act_input_1, act_input_2, M, N, K,
    stride_om, stride_im, stride_wn, dtype: 'tl.constexpr', BLOCK_M:
    'tl.constexpr', GROUP_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    BLOCK_K: 'tl.constexpr', USE_GELU: 'tl.constexpr',
    SAVE_ACTIVATION_INPUTS: 'tl.constexpr', IS_EVEN_MNK: 'tl.constexpr'):
    """
    Kernel for computing Out = activation(A x W + C)

    - Input has shape (M, K)
    - Weight 1 has shape (K, N)
    - Weight 2 has shape (K, N)
    - Output has shape (M, N)

    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % GROUP_M
    pid_n = pid % num_pid_in_group // GROUP_M
    input_block_ptr = tl.make_block_ptr(base=input, shape=(M, K), strides=(
        stride_im, 1), offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M,
        BLOCK_K), order=(1, 0))
    w1_block_ptr = tl.make_block_ptr(base=w1, shape=(K, N), strides=(1,
        stride_wn), offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K,
        BLOCK_N), order=(0, 1))
    w2_block_ptr = tl.make_block_ptr(base=w2, shape=(K, N), strides=(1,
        stride_wn), offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K,
        BLOCK_N), order=(0, 1))
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, K, BLOCK_K):
        if IS_EVEN_MNK:
            x = tl.load(input_block_ptr)
            w1_blk = tl.load(w1_block_ptr)
            w2_blk = tl.load(w2_block_ptr)
        else:
            x = tl.load(input_block_ptr, boundary_check=(0, 1))
            w1_blk = tl.load(w1_block_ptr, boundary_check=(0, 1))
            w2_blk = tl.load(w2_block_ptr, boundary_check=(0, 1))
        acc1 += tl.dot(x, w1_blk)
        acc2 += tl.dot(x, w2_blk)
        input_block_ptr = tl.advance(input_block_ptr, (0, BLOCK_K))
        w1_block_ptr = tl.advance(w1_block_ptr, (BLOCK_K, 0))
        w2_block_ptr = tl.advance(w2_block_ptr, (BLOCK_K, 0))
    if SAVE_ACTIVATION_INPUTS:
        act_in_1_ptrs = tl.make_block_ptr(base=act_input_1, shape=(M, N),
            strides=(stride_om, 1), offsets=(pid_m * BLOCK_M, pid_n *
            BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        act_in_2_ptrs = tl.make_block_ptr(base=act_input_2, shape=(M, N),
            strides=(stride_om, 1), offsets=(pid_m * BLOCK_M, pid_n *
            BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        if IS_EVEN_MNK:
            tl.store(act_in_1_ptrs, acc1)
            tl.store(act_in_2_ptrs, acc2)
        else:
            tl.store(act_in_1_ptrs, acc1, boundary_check=(0, 1))
            tl.store(act_in_2_ptrs, acc2, boundary_check=(0, 1))
    if USE_GELU:
        acc1 = gelu(acc1)
    else:
        acc1 = relu(acc1)
    acc = acc1 * acc2
    out_ptrs = tl.make_block_ptr(base=out, shape=(M, N), strides=(stride_om,
        1), offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(
        BLOCK_M, BLOCK_N), order=(1, 0))
    if IS_EVEN_MNK:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, boundary_check=(0, 1))
