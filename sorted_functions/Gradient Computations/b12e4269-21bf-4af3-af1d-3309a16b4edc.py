import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_preprocess_cumsum_gv(V, GV, GV_cumsum, DGV_cumsum_exp, DV_reduce,
    DGV_last_exp, DGV_cumsum, DV, DGV, NUM_CHUNK, L, D_MODEL_V:
    'tl.constexpr', CHUNK_SIZE: 'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    V_ptr = (V + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE *
        D_MODEL_V + tl.arange(0, D_MODEL_V))
    GV_ptr = (GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE *
        D_MODEL_V + tl.arange(0, D_MODEL_V))
    GV_cumsum_ptr = (GV_cumsum + offset_bh * L * D_MODEL_V + offset_c *
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V))
    DV_ptr = (DV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE *
        D_MODEL_V + tl.arange(0, D_MODEL_V))
    DV_reduce_ptr = (DV_reduce + offset_bh * L * D_MODEL_V + offset_c *
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V))
    DGV_cumsum_ptr = (DGV_cumsum + offset_bh * L * D_MODEL_V + offset_c *
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V))
    DGV_cumsum_exp_ptr = (DGV_cumsum_exp + offset_bh * L * D_MODEL_V + 
        offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V))
    DGV_ptr = (DGV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE *
        D_MODEL_V + tl.arange(0, D_MODEL_V))
    D_GV_last_exp_ptr = (DGV_last_exp + offset_bh * NUM_CHUNK * D_MODEL_V +
        offset_c * D_MODEL_V + tl.arange(0, D_MODEL_V))
    cumsum_gradient = tl.zeros([D_MODEL_V], dtype=tl.float32)
    grad_gv_last = tl.zeros([D_MODEL_V], dtype=tl.float32)
    gv_last = tl.load(GV_cumsum_ptr + (CHUNK_SIZE - 1) * D_MODEL_V)
    cumsum_gradient += tl.load(D_GV_last_exp_ptr) * tl.exp(gv_last)
    GV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    GV_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    V_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DV_reduce_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_cumsum_exp_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    for idx in range(CHUNK_SIZE - 1, -1, -1):
        gv_cs = tl.load(GV_cumsum_ptr)
        v = tl.load(V_ptr)
        grad_v = tl.exp(gv_last - gv_cs) * tl.load(DV_reduce_ptr)
        tl.store(DV_ptr, grad_v)
        grad_v *= v
        cumsum_gradient -= grad_v
        grad_gv_last += grad_v
        grad_v = tl.exp(gv_cs) * tl.load(DGV_cumsum_exp_ptr)
        cumsum_gradient += grad_v
        cumsum_gradient += tl.load(DGV_cumsum_ptr)
        tl.store(DGV_ptr, cumsum_gradient)
        V_ptr -= D_MODEL_V
        DV_reduce_ptr -= D_MODEL_V
        GV_cumsum_ptr -= D_MODEL_V
        DGV_cumsum_ptr -= D_MODEL_V
        DV_ptr -= D_MODEL_V
        DGV_ptr -= D_MODEL_V
        DGV_cumsum_exp_ptr -= D_MODEL_V
    DGV_ptr = (DGV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE *
        D_MODEL_V + tl.arange(0, D_MODEL_V) + (CHUNK_SIZE - 1) * D_MODEL_V)
    GV_ptr = (GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE *
        D_MODEL_V + tl.arange(0, D_MODEL_V) + (CHUNK_SIZE - 1) * D_MODEL_V)
    grad_gv_last = grad_gv_last + 0.0
    for idx in range(CHUNK_SIZE - 1, -1, -1):
        dgv = tl.load(DGV_ptr)
        dgv += grad_gv_last
        tl.store(DGV_ptr, dgv)
        DGV_ptr -= D_MODEL_V
        GV_ptr -= D_MODEL_V
