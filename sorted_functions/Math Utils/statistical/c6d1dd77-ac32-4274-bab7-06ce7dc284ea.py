import triton
import triton.language as tl
import torch

@triton.jit
def add_v8_bf16(a_hi, a_lo, b_hi, b_lo):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .v4 .b32 %acc, %tmp;
            mov.v4.b32  %acc, 0;
            mov.b64     {%acc.x, %acc.y}, $2;
            mov.b64     {%acc.z, %acc.w}, $3;
            mov.b64     {%tmp.x, %tmp.y}, $4;
            mov.b64     {%tmp.z, %tmp.w}, $5;
            add.bf16x2  %acc.x, %acc.x, %tmp.x;
            add.bf16x2  %acc.y, %acc.y, %tmp.y;
            add.bf16x2  %acc.z, %acc.z, %tmp.z;
            add.bf16x2  %acc.w, %acc.w, %tmp.w;
            mov.b64     $0, {%acc.x, %acc.y};
            mov.b64     $1, {%acc.z, %acc.w};
        }
        """
        , '=l,=l,l,l,l,l', args=[a_hi, a_lo, b_hi, b_lo], dtype=(tl.uint64,
        tl.uint64), is_pure=True, pack=1)
