import triton
import triton.language as tl
import torch

@triton.jit
def jagged_recover_first_or_last_1D_kernel(JaggedIn_no_first,
    JaggedIn_no_last, JaggedOut, OffsetsOut, BLOCK_N: 'tl.constexpr'):
    off_z = tl.program_id(0)
    group_n = tl.program_id(1)
    in_no_last_seq_start = tl.load(OffsetsOut + off_z) - off_z
    in_no_last_seq_end = tl.load(OffsetsOut + off_z + 1) - off_z - 1
    in_seq_len = in_no_last_seq_end - in_no_last_seq_start
    out_seq_start = tl.load(OffsetsOut + off_z)
    off_n = group_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_first = tl.load(JaggedIn_no_last + in_no_last_seq_start + off_n, mask
        =off_n == 0)
    tl.store(JaggedOut + out_seq_start + off_n, x_first, mask=off_n == 0)
    x = tl.load(JaggedIn_no_last + in_no_last_seq_start + off_n, mask=(
        off_n < in_seq_len) & (off_n > 0)) + tl.load(JaggedIn_no_first +
        in_no_last_seq_start + off_n - 1, mask=(off_n < in_seq_len) & (
        off_n > 0))
    tl.store(JaggedOut + out_seq_start + off_n, x, mask=(off_n < in_seq_len
        ) & (off_n > 0))
    x_last = tl.load(JaggedIn_no_first + in_no_last_seq_start + off_n, mask
        =off_n == in_seq_len - 1)
    tl.store(JaggedOut + out_seq_start + off_n + 1, x_last, mask=off_n == 
        in_seq_len - 1)
