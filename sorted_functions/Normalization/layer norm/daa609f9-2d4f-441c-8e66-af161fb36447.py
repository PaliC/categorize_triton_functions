import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'XBLOCK': 1, 'RBLOCK': 1024},
    num_stages=1, num_warps=8), triton.Config({'XBLOCK': 1, 'RBLOCK': 2048},
    num_stages=1, num_warps=8)], key=['xnumel', 'rnumel'])
@triton.jit
def triton_red_fused_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: 'tl.constexpr',
    RBLOCK: 'tl.constexpr'):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask, eviction_policy
            ='evict_last')
        tmp1 = tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = (triton_helpers.
            welford_reduce(tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0)
            )
        tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(tmp3_mean,
        tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + x0, tmp3, None)
    tmp6 = rnumel
    tmp7 = tmp4 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp10, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask,
            eviction_policy='evict_first')
        tmp15 = tl.load(in_ptr1 + r1, rmask, eviction_policy='evict_last')
        tmp18 = tl.load(in_ptr2 + r1, rmask, eviction_policy='evict_last')
        tmp12 = tmp11
        tmp13 = tmp12 - tmp3
        tmp14 = tmp13 * tmp10
        tmp16 = tmp15
        tmp17 = tmp14 * tmp16
        tmp19 = tmp18
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20
        tl.store(out_ptr1 + (r1 + rnumel * x0), tmp21, rmask)
