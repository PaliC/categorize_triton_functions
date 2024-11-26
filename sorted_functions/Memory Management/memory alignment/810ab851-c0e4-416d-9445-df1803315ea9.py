import triton
import triton.language as tl
import torch

@triton.jit
def triton_wait(wait_addrs):
    flat_tid = get_flat_tid()
    if flat_tid == 0:
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    // No need to acquire here since all threads will
                    // acquire this location after the barrier.
                    atom.global.sys.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;

                barrier_end:
            }
            """
            , '=r, l', [wait_addrs], dtype=tl.int32, is_pure=False, pack=1)
