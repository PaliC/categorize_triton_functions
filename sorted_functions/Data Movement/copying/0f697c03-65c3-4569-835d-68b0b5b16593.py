import triton
import triton.language as tl
import torch

@triton_autotune(configs=_get_configs(), key=['Z', 'H', 'N'])
@triton.heuristics({'EVEN_M': lambda args: args['N'] % args['BLOCK_M'] == 0,
    'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0})
@triton.jit
def _dense_bias_to_jagged(Z, H, N, jg_offsets_ptr, jg2_offsets_ptr,
    dense_bias_ptr, jagged_ptr, BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', EVEN_M: 'tl.constexpr', EVEN_N: 'tl.constexpr'):
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    seq_start = tl.load(jg_offsets_ptr + off_z)
    seq_end = tl.load(jg_offsets_ptr + off_z + 1)
    seq_len = seq_end - seq_start
    if start_m >= seq_len:
        return
    bias_start = tl.load(jg2_offsets_ptr + off_z
        ) * H + off_h * seq_len * seq_len
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (offs_m < seq_len)[:, None]
    off_jg_bias = bias_start + offs_m[:, None] * seq_len + offs_n[None, :]
    jg_bias_ptrs = jagged_ptr + off_jg_bias
    off_d_bias = off_hz * N * N + offs_m[:, None] * N + offs_n[None, :]
    d_bias_ptrs = dense_bias_ptr + off_d_bias
    for start_n in range(0, seq_len, BLOCK_N):
        maxk_n = (offs_n < seq_len - start_n)[None, :]
        d_bias = tl.load(d_bias_ptrs + start_n, mask=mask_m & maxk_n, other=0.0
            )
        tl.store(jg_bias_ptrs + start_n, d_bias, mask=mask_m & maxk_n)
