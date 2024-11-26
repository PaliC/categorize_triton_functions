import triton
import triton.language as tl
import torch

@triton.jit
def quantize_row_q8_triton(A, M, K, stride_am, stride_ak, stride_qm,
    stride_qk, stride_sm, Q, S, MCache: 'tl.constexpr', BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'):
    pid_m = tl.program_id(axis=0)
    A_Block_ptr = tl.make_block_ptr(base=A, shape=(M, K), block_shape=(
        BLOCK_SIZE_M, BLOCK_SIZE_K), offsets=(pid_m * BLOCK_SIZE_M, 0),
        strides=(stride_am, stride_ak), order=(0, 1))
    Q_Block_ptr = tl.make_block_ptr(base=Q, shape=(M, K), block_shape=(
        BLOCK_SIZE_M, BLOCK_SIZE_K), offsets=(pid_m * BLOCK_SIZE_M, 0),
        strides=(stride_qm, stride_qk), order=(0, 1))
    S_Block_ptr = tl.make_block_ptr(base=S, shape=(M,), block_shape=(
        BLOCK_SIZE_M,), offsets=(pid_m * BLOCK_SIZE_M,), strides=(stride_sm
        ,), order=(0,))
    a = tl.load(A_Block_ptr)
    scales = tl.max(tl.abs(a), axis=1) / 127.0
    doted = a * tl.where(scales > 0, 1 / scales, 0)[:, None]
    quant = trround(doted)
    tl.store(Q_Block_ptr, quant)
    tl.store(S_Block_ptr, scales)
