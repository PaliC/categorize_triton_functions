import triton
import triton.language as tl
import torch

@triton_autotune(configs=_get_bwd_dwdb_configs(), key=[])
@triton.jit
def _group_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, N, BLOCK_N: 'tl.constexpr'
    ):
    col = tl.program_id(0)
    num_heads = tl.num_programs(0)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        mask = rows < N
        offs = rows * num_heads + col
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + col, sum_dw)
    tl.store(FINAL_DB + col, sum_db)
