import triton
import triton.language as tl
import torch

@triton.jit
def dequantize_row_q8_triton(Q, S, M, K, stride_am, stride_ak, stride_qm,
    stride_qk, stride_sm, A, MCache: 'tl.constexpr', BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    A_Block_ptr = tl.make_block_ptr(base=A, shape=(M, K), block_shape=(
        BLOCK_SIZE_M, BLOCK_SIZE_K), offsets=(pid_m * BLOCK_SIZE_M, pid_k *
        BLOCK_SIZE_K), strides=(stride_am, stride_ak), order=(0, 1))
    Q_Block_ptr = tl.make_block_ptr(base=Q, shape=(M, K), block_shape=(
        BLOCK_SIZE_M, BLOCK_SIZE_K), offsets=(pid_m * BLOCK_SIZE_M, pid_k *
        BLOCK_SIZE_K), strides=(stride_qm, stride_qk), order=(0, 1))
    S_Block_ptr = tl.make_block_ptr(base=S, shape=(M,), block_shape=(
        BLOCK_SIZE_M,), offsets=(pid_m * BLOCK_SIZE_M,), strides=(stride_sm
        ,), order=(0,))
    quants = tl.load(Q_Block_ptr)
    scale = tl.load(S_Block_ptr)
    out = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float32)
    out += quants * scale[:, None]
    tl.store(A_Block_ptr, out)
