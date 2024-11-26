import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'XBLOCK': 1, 'RBLOCK': 2048},
    num_stages=1, num_warps=8), triton.Config({'XBLOCK': 64, 'RBLOCK': 8},
    num_stages=1, num_warps=8), triton.Config({'XBLOCK': 64, 'RBLOCK': 4},
    num_stages=1, num_warps=8), triton.Config({'XBLOCK': 8, 'RBLOCK': 512},
    num_stages=1, num_warps=8), triton.Config({'XBLOCK': 8, 'RBLOCK': 256},
    num_stages=1, num_warps=8), triton.Config({'XBLOCK': 64, 'RBLOCK': 64},
    num_stages=1, num_warps=8)], key=['xnumel', 'rnumel'])
@triton.jit
def triton_red_fused_mv_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel,
    rnumel, XBLOCK: 'tl.constexpr', RBLOCK: 'tl.constexpr'):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0 // rnumel, None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr2 + r1, None, eviction_policy='evict_last')
        tmp1 = tmp0 + 8
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tmp4 = tl.load(in_ptr1 + (r1 + rnumel * (x0 % rnumel) + rnumel *
            rnumel * tmp3), None, eviction_policy='evict_first')
        tmp5 = tmp4
        tmp6 = tmp5
        tmp8 = tmp7
        tmp9 = tmp6 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tmp12
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tmp11
    tl.store(out_ptr1 + x0, tmp13, None)
