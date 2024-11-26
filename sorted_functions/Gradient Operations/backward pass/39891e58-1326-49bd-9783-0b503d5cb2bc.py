import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_kernel_dav(V, GV, A, O, DO, DA, DV, DGV, Z, H, stride_a1,
    stride_a2, stride_a3, stride_a4, stride_v1, stride_v2, stride_v3,
    stride_v4, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    BLOCK_DMODEL_V: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_v = tl.program_id(2)
    a_offset = off_hz * stride_a2
    da_offset = (off_v * Z * H + off_hz) * stride_a2
    v_offset = off_hz * stride_v2 + off_v * BLOCK_DMODEL_V
    lo = 0
    hi = BLOCK_N
    DO_ptr = DO + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V
        )[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    O_ptr = O + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        None, :] + tl.arange(0, 16)[:, None] * stride_v4
    DV_ptr = DV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V
        )[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    GV_ptr = GV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V
        )[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    DGV_ptr = DGV + v_offset + start_m * stride_v3 + tl.arange(0,
        BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    A_ptr = A + a_offset + start_m * stride_a3 + tl.arange(0, 16)[None, :
        ] + tl.arange(0, 16)[:, None] * stride_a4
    DA_ptr = DA + da_offset + start_m * stride_a3 + tl.arange(0, 16)[None, :
        ] + tl.arange(0, 16)[:, None] * stride_a4
    for q_high in range(lo, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)
        o = tl.load(O_ptr + q_high * stride_v4)
        tl.store(DGV_ptr + q_high * stride_v4, do * o)
        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + 
            q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
        q_gv = tl.load(GV_ptr + q_high * stride_v4)
        q_gv = tl.exp(q_gv - q_gv_normalizer[None, :])
        do = do * q_gv
        tl.store(DO_ptr + q_high * stride_v4, do)
    tl.debug_barrier()
    V_ptr = V + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        :, None] + tl.arange(0, 16)[None, :] * stride_v4
    GV_ptr = GV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V
        )[:, None] + tl.arange(0, 16)[None, :] * stride_v4
    for q_high in range(lo + 16, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)
        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + 
            q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
        for k_high in range(0, q_high, 16):
            v = tl.load(V_ptr + k_high * stride_v4)
            k_gv = tl.load(GV_ptr + k_high * stride_v4)
            k_gv = tl.exp(q_gv_normalizer[:, None] - k_gv)
            v2 = v * k_gv
            dqk = tl.dot(do, v2, allow_tf32=False)
            tl.store(DA_ptr + q_high * stride_a4 + k_high, dqk)
    tl.debug_barrier()
    A_ptr = A + a_offset + start_m * stride_a3 + tl.arange(0, 16)[:, None
        ] + tl.arange(0, 16)[None, :] * stride_a4
    V_ptr = V + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[
        None, :] + tl.arange(0, 16)[:, None] * stride_v4
    GV_ptr = GV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V
        )[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    for k_high in range(0, hi, 16):
        dv = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)
        k_gv = tl.load(GV_ptr + k_high * stride_v4)
        for q_high in range(k_high + 16, BLOCK_N, 16):
            do = tl.load(DO_ptr + q_high * stride_v4)
            kq = tl.load(A_ptr + q_high * stride_a4 + k_high)
            q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 +
                q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
            k_gv2 = tl.exp(q_gv_normalizer[None, :] - k_gv)
            dv2 = tl.dot(kq, do, allow_tf32=False)
            dv += dv2 * k_gv2
        v = tl.load(V_ptr + k_high * stride_v4)
        tl.store(DV_ptr + k_high * stride_v4, dv)
        prev_dv = tl.load(DGV_ptr + k_high * stride_v4)
        tl.store(DGV_ptr + k_high * stride_v4, prev_dv - dv * v)
    tl.debug_barrier()
    A_ptr = A + a_offset + start_m * stride_a3 + tl.arange(0, 16)[:, None
        ] + tl.arange(0, 16)[None, :] * stride_a4
    for q_high in range(lo, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)
        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + 
            q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
        v = tl.load(V_ptr + q_high * stride_v4)
        k_gv = tl.load(GV_ptr + q_high * stride_v4)
        k_gv = tl.exp(q_gv_normalizer[None, :] - k_gv)
        v2 = v * k_gv
        dqk = tl.dot(do, tl.trans(v2), allow_tf32=False)
        dqk = tl.where(tl.arange(0, 16)[:, None] >= tl.arange(0, 16)[None,
            :], dqk, 0.0)
        tl.store(DA_ptr + q_high * stride_a4 + q_high, dqk)
        kq = tl.load(A_ptr + q_high * stride_a4 + q_high)
        dv2 = tl.dot(kq, do, allow_tf32=False)
        dv = dv2 * k_gv
        prev_dv = tl.load(DV_ptr + q_high * stride_v4)
        tl.store(DV_ptr + q_high * stride_v4, prev_dv + dv)
        prev_gdv = tl.load(DGV_ptr + q_high * stride_v4)
        prev_gdv -= dv * v
        tl.store(DGV_ptr + q_high * stride_v4, prev_gdv)
