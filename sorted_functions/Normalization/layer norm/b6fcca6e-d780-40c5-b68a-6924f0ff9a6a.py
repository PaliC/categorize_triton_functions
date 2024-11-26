import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'XBLOCK': 1, 'RBLOCK': 1024},
    num_stages=1, num_warps=8), triton.Config({'XBLOCK': 1, 'RBLOCK': 2048},
    num_stages=1, num_warps=8)], key=['xnumel', 'rnumel'])
@triton.jit
def triton_red_fused_native_layer_norm_no_welford(in_out_ptr0, in_out_ptr1,
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK:
    'tl.constexpr', RBLOCK: 'tl.constexpr'):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask, eviction_policy
            ='evict_last')
        tmp1 = tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tmp4
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = rnumel
    tmp6 = tmp3 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp6, None)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask, eviction_policy
            ='evict_last')
        tmp8 = tmp7
        tmp9 = tmp8 - tmp6
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tmp13
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp14 = rnumel
    tmp15 = tmp12 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + x0, tmp18, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(in_ptr0 + (r1 + rnumel * x0), rmask,
            eviction_policy='evict_first')
        tmp23 = tl.load(in_ptr1 + r1, rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr2 + r1, rmask, eviction_policy='evict_last')
        tmp20 = tmp19
        tmp21 = tmp20 - tmp6
        tmp22 = tmp21 * tmp18
        tmp24 = tmp23
        tmp25 = tmp22 * tmp24
        tmp27 = tmp26
        tmp28 = tmp25 + tmp27
        tmp29 = tmp28
        tl.store(out_ptr0 + (r1 + rnumel * x0), tmp29, rmask)
