import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_bwd_dwdb(A, DOut, Mean, Var, DW, DB, M, N, BLOCK_SIZE_M:
    'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    UNROLL: 'tl.constexpr' = 4
    for i in range(0, M, BLOCK_SIZE_M * UNROLL):
        for j in range(UNROLL):
            rows = i + j * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            a = tl.load(A + offs, mask=mask, other=0.0)
            dout = tl.load(DOut + offs, mask=mask, other=0.0)
            mean = tl.load(Mean + rows, mask=rows < M, other=0.0)
            rstd = tl.load(Var + rows, mask=rows < M, other=0.0)
            a_hat = (a - mean[:, None]) * rstd[:, None]
            dw += dout * a_hat
            db += dout
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(DW + cols, sum_dw, mask=cols < N)
    tl.store(DB + cols, sum_db, mask=cols < N)
