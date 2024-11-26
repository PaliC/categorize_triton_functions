import triton
import triton.language as tl
import torch

@triton.jit
def triton_send(send_addrs):
    flat_tid = get_flat_tid()
    if flat_tid == 0:
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;

                barrier_end:
            }
            """
            , '=r, l', [send_addrs], dtype=tl.int32, is_pure=False, pack=1)
