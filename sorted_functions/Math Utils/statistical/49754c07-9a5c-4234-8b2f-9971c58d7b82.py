import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BS': 32}, num_warps=2), triton.
    Config({'BS': 32}, num_warps=4), triton.Config({'BS': 32}, num_warps=8),
    triton.Config({'BS': 64}, num_warps=2), triton.Config({'BS': 64},
    num_warps=4), triton.Config({'BS': 64}, num_warps=8), triton.Config({
    'BS': 128}, num_warps=2), triton.Config({'BS': 128}, num_warps=4),
    triton.Config({'BS': 128}, num_warps=8)], key=['S'])
@triton.jit
def recurrent_cumsum_bwd_kernel(ds, dz, s_s_h, s_s_t, T: 'tl.constexpr', S:
    'tl.constexpr', BS: 'tl.constexpr'):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_s = i_s * BS + tl.arange(0, BS)
    mask = o_s < S
    b_ds = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(T - 1, -1, -1):
        b_dz = tl.load(dz + i_bh * s_s_h + i_t * s_s_t + o_s, mask=mask,
            other=0)
        b_ds = b_ds + b_dz
        tl.store(ds + i_bh * s_s_h + i_t * s_s_t + o_s, b_ds, mask=mask)