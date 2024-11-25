import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_init_att_window_info(b_seq_len, b_att_seq_len, batch_size,
    sliding_window, BLOCK_SIZE: 'tl.constexpr'):
    cur_index = tl.program_id(0)
    cur_start = cur_index * BLOCK_SIZE
    offsets = cur_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    cur_seq_len = tl.load(b_seq_len + offsets, mask=mask)
    b_att_seq_len_data = tl.minimum(cur_seq_len, sliding_window)
    tl.store(b_att_seq_len + offsets, b_att_seq_len_data, mask=mask)
    return
