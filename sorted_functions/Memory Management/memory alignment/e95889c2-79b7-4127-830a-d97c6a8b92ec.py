import triton
import triton.language as tl
import torch

@triton.jit
def load_128(addrs, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32             %p0, $3, 1;
            @%p0 ld.global.v2.u64   {$0, $1}, [$2];
        }
        """
        , '=l,=l,l,r', args=[addrs, mask], dtype=(tl.uint64, tl.uint64),
        is_pure=True, pack=1)
