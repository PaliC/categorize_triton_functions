import triton
import triton.language as tl
import torch

@triton.jit
def qkv_proj(x_ptr, q_weight_ptr, k_weight_ptr, v_weight_ptr, q_ptr, k_ptr,
    v_ptr, M, N, K, stride_x_batch, stride_x_m, stride_x_k, stride_q_w_k,
    stride_q_w_n, stride_k_w_k, stride_k_w_n, stride_v_w_k, stride_v_w_n,
    stride_q_batch, stride_q_m, stride_q_n, stride_k_batch, stride_k_m,
    stride_k_n, stride_v_batch, stride_v_m, stride_v_n, USE_FP8:
    'tl.constexpr', EPS: 'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr',
    BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'):
    matmul_(x_ptr=x_ptr, w_ptr=q_weight_ptr, out_ptr=q_ptr, M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=
        stride_x_k, stride_w_k=stride_q_w_k, stride_w_n=stride_q_w_n,
        stride_out_batch=stride_q_batch, stride_out_m=stride_q_m,
        stride_out_n=stride_q_n, USE_FP8=USE_FP8, EPS=EPS, BLOCK_SIZE_M=
        BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K)
    matmul_(x_ptr=x_ptr, w_ptr=k_weight_ptr, out_ptr=k_ptr, M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=
        stride_x_k, stride_w_k=stride_k_w_k, stride_w_n=stride_k_w_n,
        stride_out_batch=stride_k_batch, stride_out_m=stride_k_m,
        stride_out_n=stride_k_n, USE_FP8=USE_FP8, EPS=EPS, BLOCK_SIZE_M=
        BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K)
    matmul_(x_ptr=x_ptr, w_ptr=v_weight_ptr, out_ptr=v_ptr, M=M, N=N, K=K,
        stride_x_batch=stride_x_batch, stride_x_m=stride_x_m, stride_x_k=
        stride_x_k, stride_w_k=stride_v_w_k, stride_w_n=stride_v_w_n,
        stride_out_batch=stride_v_batch, stride_out_m=stride_v_m,
        stride_out_n=stride_v_n, USE_FP8=USE_FP8, EPS=EPS, BLOCK_SIZE_M=
        BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K)
