import triton
import triton.language as tl
import torch

@triton.jit
def blockwise_barrier_double_tree(signal_pad_ptrs, block_id, send1_rank,
    send2_rank, wait1_rank, wait2_rank, RANK: 'tl.constexpr', WORLD_SIZE:
    'tl.constexpr'):
    if block_id is None:
        block_id = tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0
            ) + tl.program_id(1) * tl.num_programs(0) + tl.program_id(0)
    flat_tid = get_flat_tid()
    remote_ranks = tl.cat(tl.full((1,), send1_rank, tl.int32), tl.full((1,),
        send2_rank, tl.int32))
    signal_pad_ptrs = signal_pad_ptrs
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks)
    send_addrs = remote_signal_pad_addrs + block_id * WORLD_SIZE + RANK
    remote_ranks = tl.cat(tl.full((1,), wait1_rank, tl.int32), tl.full((1,),
        wait2_rank, tl.int32))
    local_signal_pad_addr = tl.load(signal_pad_ptrs + RANK)
    wait_addrs = local_signal_pad_addr + block_id * WORLD_SIZE + remote_ranks
    if flat_tid < WORLD_SIZE:
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;

                wait_signal:
                    // No need to acquire here since all threads will
                    // acquire this location after the barrier.
                    atom.global.sys.cas.b32 %tmp32_0, [$2], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;

                barrier_end:
            }
            """
            , '=r, l, l', [send_addrs, wait_addrs], dtype=tl.int32, is_pure
            =False, pack=1)
    tl.inline_asm_elementwise('bar.sync 0;', '=r', [], dtype=tl.int32,
        is_pure=False, pack=1)
    tl.inline_asm_elementwise('ld.acquire.sys.global.u32 $0, [$1];',
        '=r, l', [local_signal_pad_addr + send1_rank], dtype=tl.int32,
        is_pure=False, pack=1)
    tl.inline_asm_elementwise('ld.acquire.sys.global.u32 $0, [$1];',
        '=r, l', [local_signal_pad_addr + send2_rank], dtype=tl.int32,
        is_pure=False, pack=1)
