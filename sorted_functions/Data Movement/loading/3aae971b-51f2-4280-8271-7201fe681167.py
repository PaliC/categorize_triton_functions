import triton
import triton.language as tl
import torch

@triton.jit
def patching_kernel(image_ptr, out_ptr, batch_size, batch_stride,
    image_stride0, image_stride1, N, H, W, C, P, block_x: 'tl.constexpr',
    block_y: 'tl.constexpr'):
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)
    col_idx = tl.program_id(axis=2)
    batch_offset = batch_idx * batch_stride
    row_offset = row_idx * block_y + tl.arange(0, block_y)
    col_offset = col_idx * block_x + tl.arange(0, block_x)
    data_offset = row_offset[None, :] * image_stride1 + col_offset[:, None]
    row_mask = row_offset < H
    col_mask = col_offset < W
    data_mask = row_mask[:, None] & col_mask[None, :]
    img_r = tl.load(image_ptr + batch_offset + data_offset, mask=data_mask)
    img_g = tl.load(image_ptr + batch_offset + data_offset + image_stride0,
        mask=data_mask)
    img_b = tl.load(image_ptr + batch_offset + data_offset + image_stride0 *
        2, mask=data_mask)
    P_single_row = P * P * C
    num_patches_x = (W + P - 1) // P
    P_offset = (row_idx * num_patches_x + col_idx) * P_single_row
    out_offset = P_offset + tl.arange(0, block_x * block_y)
    out_mask = out_offset < N * P * P * C
    tl.store(out_ptr + batch_offset + out_offset, tl.ravel(img_r), mask=
        out_mask)
    tl.store(out_ptr + batch_offset + out_offset + P * P, tl.ravel(img_g),
        mask=out_mask)
    tl.store(out_ptr + batch_offset + out_offset + P * P * 2, tl.ravel(
        img_b), mask=out_mask)
