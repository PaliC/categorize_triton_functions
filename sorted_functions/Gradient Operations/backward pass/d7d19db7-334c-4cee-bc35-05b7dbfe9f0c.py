import triton
import triton.language as tl
import torch

@triton_autotune(configs=_get_bwd_dwdb_configs(), key=['D'])
@triton.jit
def _ln_mul_dropout_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, N, D, BLOCK_N:
    'tl.constexpr', BLOCK_D: 'tl.constexpr'):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < D)
