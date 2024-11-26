import triton
import triton.language as tl
import torch

@triton.autotune(config_gen(), key=['N_BLK', 'BLK1_IN', 'BLK2_OUT'])
@triton.jit
def monarch_backward(dout_ptr, out1_ptr, x_ptr, w1_bfly_ptr, w2_bfly_ptr,
    dx_ptr, dw1_bfly_ptr, dw2_bfly_ptr, SEQ_DIM, N_BLK, BLK1_IN, BLK1_OUT,
    BLK2_OUT, stride_dout_l, stride_dout_m, stride_dout_n, stride_out1_r,
    stride_out1_m, stride_out1_l, stride_xl, stride_xm, stride_xk,
    stride_w1l, stride_w1r, stride_w1k, stride_w2l, stride_w2n, stride_w2r,
    BLOCK_SIZE_SEQ: 'tl.constexpr'=64, BLOCK_SIZE_N: 'tl.constexpr'=64,
    BLOCK_SIZE_K: 'tl.constexpr'=32, GROUP_SIZE_M: 'tl.constexpr'=8):
    BLK2_IN = BLK1_OUT
    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)
    num_pid_m = tl.cdiv(SEQ_DIM, BLOCK_SIZE_SEQ)
    num_pid_n = tl.cdiv(N_BLK * BLK1_IN, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = pid_m * BLOCK_SIZE_SEQ
    offs_n = pid_n * BLOCK_SIZE_N
    offs_k = 0
    x_ptrs = tl.make_block_ptr(x_ptr + pid_batch * stride_xl, shape=(
        SEQ_DIM, BLK1_IN), strides=(stride_xm, stride_xk), offsets=(offs_m,
        offs_k), block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_K), order=(0, 1))
    dx_ptrs = tl.make_block_ptr(dx_ptr + pid_batch * stride_xl, shape=(
        SEQ_DIM, BLK1_IN), strides=(stride_xm, stride_xk), offsets=(offs_m,
        offs_k), block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_K), order=(1, 0))
    out1_ptrs = tl.make_block_ptr(out1_ptr + pid_batch * stride_out1_l,
        shape=(SEQ_DIM, BLK1_OUT), strides=(stride_out1_m, stride_out1_r),
        offsets=(offs_m, 0), block_shape=(BLOCK_SIZE_SEQ, BLK1_OUT), order=
        (1, 0))
    dout_ptrs = tl.make_block_ptr(dout_ptr + pid_batch * stride_dout_l,
        shape=(SEQ_DIM, BLK2_OUT), strides=(stride_dout_m, stride_dout_n),
        offsets=(offs_m, offs_n), block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_N
        ), order=(1, 0))
    w1_ptrs = tl.make_block_ptr(w1_bfly_ptr + pid_batch * stride_w1l, shape
        =(BLK1_OUT, BLK1_IN), strides=(stride_w1r, stride_w1k), offsets=(0,
        offs_k), block_shape=(BLK1_OUT, BLOCK_SIZE_K), order=(1, 0))
    dw1_ptrs = tl.make_block_ptr(dw1_bfly_ptr + pid_batch * stride_w1l,
        shape=(BLK1_OUT, BLK1_IN), strides=(stride_w1r, stride_w1k),
        offsets=(0, offs_k), block_shape=(BLK1_OUT, BLOCK_SIZE_K), order=(1, 0)
        )
    w2_ptrs = tl.make_block_ptr(w2_bfly_ptr + pid_batch * stride_w2l, shape
        =(BLK2_OUT, BLK2_IN), strides=(stride_w2n, stride_w2r), offsets=(
        offs_n, 0), block_shape=(BLOCK_SIZE_N, BLK2_IN), order=(1, 0))
    dw2_ptrs = tl.make_block_ptr(dw2_bfly_ptr + pid_batch * stride_w2l,
        shape=(BLK2_OUT, BLK2_IN), strides=(stride_w2n, stride_w2r),
        offsets=(offs_n, 0), block_shape=(BLOCK_SIZE_N, BLK2_IN), order=(1, 0))
    dout = tl.load(dout_ptrs, boundary_check=(0, 1), eviction_policy=
        'evict_first')
    out1 = tl.load(out1_ptrs, boundary_check=(1,), eviction_policy=
        'evict_first')
    w2_bfly = tl.load(w2_ptrs, boundary_check=(0,))
    dw2_bfly = tl.dot(tl.trans(out1), dout)
    tl.store(dw2_ptrs, dw2_bfly, boundary_check=(0,))
    x = tl.load(x_ptrs, boundary_check=(0, 1))
    dout1 = tl.dot(dout, w2_bfly)
    dx = tl.zeros((BLOCK_SIZE_SEQ, BLOCK_SIZE_K), dtype=tl.float32)
    for k in range(BLOCK_SIZE_K, BLK1_IN, BLOCK_SIZE_K):
        w1_bfly = tl.load(w1_ptrs, boundary_check=(1,))
        dx += tl.dot(dout1, w1_bfly)
        tl.advance(w1_ptrs, (0, BLOCK_SIZE_K))
    tl.store(dx_ptrs, dx, boundary_check=(0, 1))
    dw1_bfly = tl.dot(tl.trans(dout1), x)
    tl.store(dw1_ptrs, dw1_bfly, boundary_check=(1,))
