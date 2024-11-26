import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_preprocess_cumsum_gv(V, GV, GV_cumsum, GV_exp, V_reduce,
    GV_last_exp, NUM_CHUNK, L, D_MODEL_V: 'tl.constexpr', CHUNK_SIZE:
    'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    GV_ptr = (GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE *
        D_MODEL_V + tl.arange(0, D_MODEL_V))
    GV_last_exp_ptr = (GV_last_exp + offset_bh * NUM_CHUNK * D_MODEL_V + 
        offset_c * D_MODEL_V + tl.arange(0, D_MODEL_V))
    GV_cumsum_ptr = (GV_cumsum + offset_bh * L * D_MODEL_V + offset_c *
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V))
    GV_exp_ptr = (GV_exp + offset_bh * L * D_MODEL_V + offset_c *
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V))
    cumsum = tl.zeros([D_MODEL_V], dtype=tl.float32)
    for _ in range(CHUNK_SIZE):
        gv = tl.load(GV_ptr)
        cumsum += gv
        tl.store(GV_cumsum_ptr, cumsum)
        tl.store(GV_exp_ptr, tl.exp(cumsum))
        GV_cumsum_ptr += D_MODEL_V
        GV_exp_ptr += D_MODEL_V
        GV_ptr += D_MODEL_V
    tl.store(GV_last_exp_ptr, tl.exp(cumsum))
    tl.debug_barrier()
    V_ptr = (V + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE *
        D_MODEL_V + tl.arange(0, D_MODEL_V))
    GV_cumsum_ptr = (GV_cumsum + offset_bh * L * D_MODEL_V + offset_c *
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V))
    V_reduce_ptr = (V_reduce + offset_bh * L * D_MODEL_V + offset_c *
        CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V))
    for _ in range(CHUNK_SIZE):
        v = tl.load(V_ptr)
        gv = tl.load(GV_cumsum_ptr)
        v_reduce = v * tl.exp(cumsum - gv)
        tl.store(V_reduce_ptr, v_reduce)
        V_ptr += D_MODEL_V
        V_reduce_ptr += D_MODEL_V
        GV_cumsum_ptr += D_MODEL_V
