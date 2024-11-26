import triton
import triton.language as tl
import torch

@triton.jit
def add_v4_bf16(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .v4 .b32 %acc, %tmp;
            mov.v4.b32  %acc, 0;
            mov.b64     {%acc.x, %acc.y}, $1;
            mov.b64     {%tmp.x, %tmp.y}, $2;
            add.bf16x2  %acc.x, %acc.x, %tmp.x;
            add.bf16x2  %acc.y, %acc.y, %tmp.y;
            mov.b64     $0, {%acc.x, %acc.y};
        }
        """
        , '=l,l,l', args=[a, b], dtype=tl.uint64, is_pure=True, pack=1)
