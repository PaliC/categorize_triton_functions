import triton
import triton.language as tl
import torch

@triton.jit
def gated_matmul_bwd_ygrad(dout, y1_grad, y2_grad, act_input_1, act_input_2,
    M, N, stride_dom, dtype: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', USE_GELU: 'tl.constexpr', IS_EVEN_MNK:
    'tl.constexpr'):
    """
    Kernel for backward gated MLP

    Ref :
    y2_grad = torch.mul(gelu(x @ w1), dout)
    y1_grad = torch.mul(gelu_grad(x @ w1) * (x @ w2), dout)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    actin_1_block_ptr = tl.make_block_ptr(base=act_input_1, shape=(M, N),
        strides=(stride_dom, 1), offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    actin_2_block_ptr = tl.make_block_ptr(base=act_input_2, shape=(M, N),
        strides=(stride_dom, 1), offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    dout_block_ptr = tl.make_block_ptr(base=dout, shape=(M, N), strides=(
        stride_dom, 1), offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    if IS_EVEN_MNK:
        dout_blk = tl.load(dout_block_ptr)
        actin_1_blk = tl.load(actin_1_block_ptr)
        actin_2_blk = tl.load(actin_2_block_ptr)
    else:
        dout_blk = tl.load(dout_block_ptr, boundary_check=(0, 1))
        actin_1_blk = tl.load(actin_1_block_ptr, boundary_check=(0, 1))
        actin_2_blk = tl.load(actin_2_block_ptr, boundary_check=(0, 1))
    if USE_GELU:
        actin_act = gelu(actin_1_blk)
        actin_act_grad = gelu_grad(actin_1_blk)
    else:
        actin_act = relu(actin_1_blk)
        actin_act_grad = relu_grad(actin_1_blk)
    actin_act *= dout_blk
    actin_act_grad *= actin_2_blk
    actin_act_grad *= dout_blk
    y1_grad_ptrs = tl.make_block_ptr(base=y1_grad, shape=(M, N), strides=(
        stride_dom, 1), offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    y2_grad_ptrs = tl.make_block_ptr(base=y2_grad, shape=(M, N), strides=(
        stride_dom, 1), offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    if IS_EVEN_MNK:
        tl.store(y1_grad_ptrs, actin_act_grad)
        tl.store(y2_grad_ptrs, actin_act)
    else:
        tl.store(y1_grad_ptrs, actin_act_grad, boundary_check=(0, 1))
        tl.store(y2_grad_ptrs, actin_act, boundary_check=(0, 1))
