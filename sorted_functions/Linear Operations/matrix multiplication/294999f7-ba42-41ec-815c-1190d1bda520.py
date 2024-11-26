import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_compute_O(A, V, GV, O, stride_a2, stride_a3, stride_a4, stride_v2,
    stride_v3, stride_v4, BLOCK_N: 'tl.constexpr', BLOCK_DMODEL_V:
    'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_v = tl.program_id(2)
    a_offset = off_hz * stride_a2
    v_offset = off_hz * stride_v2 + off_v * BLOCK_DMODEL_V
    lo = 0
    hi = BLOCK_N
    V_ptr = V + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        None, :] + tl.arange(0, 16)[:, None] * stride_v4
    O_ptr = O + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        None, :] + tl.arange(0, 16)[:, None] * stride_v4
    GV_ptr = GV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V
        )[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    A_ptr = A + a_offset + start_m * stride_a3 + tl.arange(0, 16)[None, :
        ] + tl.arange(0, 16)[:, None] * stride_a4
    for q_high in range(lo + 16, hi, 16):
        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + 
            q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
        acc = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)
        for k_high in range(0, q_high, 16):
            qk = tl.load(A_ptr + q_high * stride_a4 + k_high)
            v = tl.load(V_ptr + k_high * stride_v4)
            k_gv = tl.load(GV_ptr + k_high * stride_v4)
            k_gv = tl.exp(q_gv_normalizer[None, :] - k_gv)
            v = v * k_gv
            output = tl.dot(qk, v, allow_tf32=False)
            acc += output
        tl.store(O_ptr + q_high * stride_v4, acc)
    tl.store(O_ptr, tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32))
    tl.debug_barrier()
    for q_high in range(lo, hi, 16):
        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + 
            q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
        qk = tl.load(A_ptr + q_high * stride_a4 + q_high)
        v = tl.load(V_ptr + q_high * stride_v4)
        k_gv = tl.load(GV_ptr + q_high * stride_v4)
        k_gv2 = tl.exp(q_gv_normalizer[None, :] - k_gv)
        v = v * k_gv2
        output = tl.dot(qk, v, allow_tf32=False)
        q_gv = tl.exp(k_gv - q_gv_normalizer[None, :])
        prev = tl.load(O_ptr + q_high * stride_v4)
        output += prev
        output = output * q_gv
        tl.store(O_ptr + q_high * stride_v4, output)
