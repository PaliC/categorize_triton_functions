import triton
import triton.language as tl
import torch

@triton.autotune(config_gen(), key=['N_BLK', 'BLK1_IN', 'BLK2_OUT'], rep=80,
    warmup=15)
@triton.jit
def monarch_forward(x_ptr, o_ptr1, o_ptr2, w1_bfly_ptr, w2_bfly_ptr,
    SEQ_DIM, N_BLK, BLK1_IN, BLK1_OUT: 'tl.constexpr', BLK2_OUT:
    'tl.constexpr', stride_xl, stride_xm, stride_xk, stride_w1l, stride_w1r,
    stride_w1k, stride_w2l, stride_w2n, stride_w2r, stride_o1l, stride_o1m,
    stride_o1k, stride_o2l, stride_o2m, stride_o2n, BLOCK_SIZE_SEQ:
    'tl.constexpr'=64, BLOCK_SIZE_N: 'tl.constexpr'=64, BLOCK_SIZE_K:
    'tl.constexpr'=32, GROUP_SIZE_M: 'tl.constexpr'=8):
    """
    Implements fused monarch forward as in `BlockdiagButterflyMultiply`.
    """
    BLK2_IN: 'tl.constexpr' = BLK1_OUT
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
    offs_am = pid_m * BLOCK_SIZE_SEQ
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    x_ptrs = tl.make_block_ptr(x_ptr + pid_batch * stride_xl, shape=(
        SEQ_DIM, BLK1_IN), strides=(stride_xm, stride_xk), offsets=(offs_am,
        offs_k), block_shape=(BLOCK_SIZE_SEQ, BLOCK_SIZE_K), order=(1, 0))
    w1_ptrs = tl.make_block_ptr(w1_bfly_ptr + pid_batch * stride_w1l, shape
        =(BLK1_OUT, BLK1_IN), strides=(stride_w1r, stride_w1k), offsets=(0,
        offs_k), block_shape=(BLK1_OUT, BLOCK_SIZE_K), order=(1, 0))
    w2_ptrs = tl.make_block_ptr(w2_bfly_ptr + pid_batch * stride_w2l, shape
        =(BLK2_OUT, BLK2_IN), strides=(stride_w2n, stride_w2r), offsets=(
        offs_bn, 0), block_shape=(BLOCK_SIZE_N, BLK2_IN), order=(1, 0))
    out1_ptrs = tl.make_block_ptr(o_ptr1 + pid_batch * stride_o1l, shape=(
        SEQ_DIM, BLK1_OUT), strides=(stride_o1m, stride_o1k), offsets=(
        offs_am, offs_k), block_shape=(BLOCK_SIZE_SEQ, BLK1_OUT), order=(1, 0))
    out2_ptrs = tl.make_block_ptr(o_ptr2, shape=(SEQ_DIM, N_BLK, BLK2_OUT),
        strides=(stride_o2l, stride_o2m, stride_o2n), offsets=(offs_am,
        pid_batch, offs_bn), block_shape=(BLOCK_SIZE_SEQ, 1, BLOCK_SIZE_N),
        order=(2, 1, 0))
    offs_am = pid_m * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x = tl.load(x_ptrs, boundary_check=(0, 1), eviction_policy=
        'evict_first', padding_option='zero')
    dtype = x.dtype
    out1 = tl.zeros((BLOCK_SIZE_SEQ, BLK1_OUT), dtype=tl.float16 if dtype ==
        tl.float16 else tl.float32)
    for k in range(0, BLK1_IN, BLOCK_SIZE_K):
        w1_bfly = tl.load(w1_ptrs, boundary_check=(0, 1), eviction_policy=
            'evict_first', padding_option='zero')
        w1_bfly = tl.trans(w1_bfly)
        out1 += tl.dot(x, w1_bfly, out_dtype=tl.float16 if dtype == tl.
            float16 else tl.float32)
        x_ptrs = tl.advance(x_ptrs, (0, BLOCK_SIZE_K))
        w1_ptrs = tl.advance(w1_ptrs, (0, BLOCK_SIZE_K))
        x = tl.load(x_ptrs, boundary_check=(0, 1), eviction_policy=
            'evict_first', padding_option='zero')
    out1 = out1
    tl.store(out1_ptrs, out1, boundary_check=(0,))
    w2_bfly = tl.load(w2_ptrs, boundary_check=(0,), padding_option='zero')
    w2_bfly = tl.trans(w2_bfly)
    out2 = tl.dot(out1, w2_bfly)
    tl.store(out2_ptrs, out2[:, None, :], boundary_check=(0, 2))
