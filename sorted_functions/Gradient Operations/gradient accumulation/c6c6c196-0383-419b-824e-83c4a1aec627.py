import triton
import triton.language as tl
import torch

@triton.jit
def _rms_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    """
    Kernel invocation for backward pass of RMS normalization, computing and aggregating gradients w.r.t. weights and biases

    Params:
        - DW (tensor): Intermediate gradient tensor for the scale factors, W
        - DB (tensor): Intermediate gradient tensor for the biases, B
        - FINAL_DW (tensor): Aggregated gradient tensor for the scale factors, to be updated
        - FINAL_DB (tensor): Aggregated gradient tensor for the biases, to be updated
        - M (int): Number of groups or batch size dimension
        - N (int): Dimensionality of the feature vectors or the number of features
        - BLOCK_SIZE_M (constexpr): Compile-time constant defining the block size in the M dimension
        - BLOCK_SIZE_N (constexpr): Compile-time constant defining the block size in the N dimension

    Return:
        - None

    Usage:
        _rms_norm_bwd_dwdb[grid, block](DW, DB, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    """
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)
