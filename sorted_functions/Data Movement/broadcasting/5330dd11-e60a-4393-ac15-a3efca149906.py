import triton
import triton.language as tl
import torch

@triton.jit
def double_tree_all_reduce_kernel(buffer_ptrs, signal_pad_ptrs, output_ptr,
    tree0_parent, tree0_child0, tree0_child1, tree1_parent, tree1_child0,
    tree1_child1, numel: 'tl.constexpr', rank: 'tl.constexpr', world_size:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', NUMEL_PER_THREAD:
    'tl.constexpr'):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size)
    block_id = tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0
        ) + tl.program_id(1) * tl.num_programs(0) + tl.program_id(0)
    pid = tl.program_id(axis=0)
    buffer_ptrs = buffer_ptrs
    output_ptr = output_ptr
    signal_pad_ptrs = signal_pad_ptrs
    block_start = pid * BLOCK_SIZE
    if tree0_child0 != -1 and tree0_child0 < 8:
        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank)
        wait_addrs = (local_signal_pad_addr + block_id * world_size +
            tree0_child0)
        triton_wait(wait_addrs)
    if tree0_child1 != -1 and tree0_child1 < 8:
        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank)
        wait_addrs = (local_signal_pad_addr + block_id * world_size +
            tree0_child1)
        triton_wait(wait_addrs)
    while block_start < numel // NUMEL_PER_THREAD:
        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE
            ) < numel // NUMEL_PER_THREAD
        acc_hi = tl.zeros((BLOCK_SIZE,), tl.uint64)
        acc_lo = tl.zeros((BLOCK_SIZE,), tl.uint64)
        if tree0_child0 != -1 and tree0_child0 < 8:
            if block_id == 0:
                if rank == 0:
                    if block_start == pid * BLOCK_SIZE:
                        buffer_ptr = tl.load(buffer_ptrs + tree0_child0)
                        None
                        hi, lo = load_128(buffer_ptr + offsets, mask=tl.
                            full((1,), 4294967295, dtype=tl.uint32))
        buffer_ptr = tl.load(buffer_ptrs + rank)
        hi, lo = load_128(buffer_ptr + offsets, mask=mask)
        acc_hi, acc_lo = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        tl.store(buffer_ptr + offsets + 0, acc_hi, mask=mask)
        tl.store(buffer_ptr + offsets + 1, acc_lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE
    if tree0_parent != -1:
        remote_signal_pad_addrs = tl.load(signal_pad_ptrs + tree0_parent)
        send_addrs = remote_signal_pad_addrs + block_id * world_size + rank
        triton_send(send_addrs)
        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank)
        wait_addrs = (local_signal_pad_addr + block_id * world_size +
            tree0_parent)
        triton_wait(wait_addrs)
