import triton
import triton.language as tl
import torch

@triton.jit
def get_flat_tid():
    return tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<2>;

            mov.u32     %tmp32_0, %tid.z;
            mov.u32     %tmp32_1, %ntid.y;
            mul.lo.u32  %tmp32_0, %tmp32_0, %tmp32_1; // tid.z * ntid.y
            mov.u32     %tmp32_1, %ntid.x;
            mul.lo.u32  $0, %tmp32_0, %tmp32_1;       // $0 = tid.z * ntid.y * ntid.x
            mov.u32     %tmp32_0, %tid.y;
            mov.u32     %tmp32_1, %ntid.x;
            mul.lo.u32  %tmp32_0, %tmp32_0, %tmp32_1; // tid.y * ntid.x
            add.u32     $0, $0, %tmp32_0;             // $0 += tid.y * ntid.x
            mov.u32     %tmp32_0, %tid.x;
            add.u32     $0, $0, %tmp32_0;             // $0 += tid.x
        }
        """
        , '=r', [], dtype=tl.int32, is_pure=True, pack=1)
