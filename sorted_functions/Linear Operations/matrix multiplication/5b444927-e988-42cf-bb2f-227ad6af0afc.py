import triton
import triton.language as tl
import torch

@triton.jit
def conv2d_kernel(x_ptr, k_ptr, z_ptr, N0, H, W, KH: 'tl.constexpr', KW:
    'tl.constexpr', B0: 'tl.constexpr'):
    block_id_i = tl.program_id(0)
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0
    off_h = tl.arange(0, KH)
    off_w = tl.arange(0, KW)
    off_hw = off_h[:, None] * KW + off_w[None, :]
    k = tl.load(k_ptr + off_hw)
    for j in tl.range(0, H):
        for l in tl.range(0, W):
            off_j_oj = j + off_h[None, :, None]
            off_l_ol = l + off_w[None, None, :]
            off_x = off_i * H * W + off_j_oj * W + off_l_ol
            mask_x = (off_j_oj < H) & (off_l_ol < W)
            x = tl.load(x_ptr + off_x, mask=mask_x)
            z = tl.sum(x * k[None, :])
            off_z = off_i * H * W + j * W + l
            tl.store(z_ptr + off_z, z)
    return
