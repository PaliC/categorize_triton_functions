import triton
import triton.language as tl
import torch

@conv_heuristics()
@triton.jit
def _kernel_delta_x_hwc(x, w, y, stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww, stride_yn, stride_yc,
    stride_yh, stride_yw, stride_biasn, delta_xh_ptr, delta_xw_ptr,
    delta_xc_ptr, BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W,
    OUT_H, OUT_W, stride_h, stride_w, padding_h, padding_w, dilation_h,
    dilation_w, output_padding_h, output_padding_w, groups, ACC_TYPE:
    'tl.constexpr', CONV1X1_NHWC: 'tl.constexpr', BLOCK_M: 'tl.constexpr',
    BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', GROUP_H: 'tl.constexpr'):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)
    CRS = IN_C * KERNEL_H * KERNEL_W
    if not CONV1X1_NHWC:
        delta_xh_ptrs = delta_xh_ptr + off_x_crs
        delta_xw_ptrs = delta_xw_ptr + off_x_crs
        delta_xc_ptrs = delta_xc_ptr + off_x_crs
        delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
        delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
        delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
        off_x_crs_unpacked = (delta_xh * stride_xh + delta_xw * stride_xw +
            delta_xc * stride_xc)
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    else:
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]
        delta_xh = 0
        delta_xw = 0
    mask_x = (off_x_n < BATCH)[:, None] & (off_x_crs < CRS)[None, :] & (
        off_x_h[:, None] + delta_xh[None, :] >= 0) & (off_x_h[:, None] +
        delta_xh[None, :] < IN_H) & (off_x_w[:, None] + delta_xw[None, :] >= 0
        ) & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):
        acc += tl.dot(matrix_x, matrix_w)
        w_ptrs += BLOCK_K
        off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
        if not CONV1X1_NHWC:
            delta_xh_ptrs += BLOCK_K
            delta_xw_ptrs += BLOCK_K
            delta_xc_ptrs += BLOCK_K
            delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
            off_x_crs_unpacked = (delta_xh * stride_xh + delta_xw *
                stride_xw + delta_xc * stride_xc)
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            x_ptrs += BLOCK_K
        mask_x = (off_x_n < BATCH)[:, None] & (off_x_crs < CRS)[None, :] & (
            off_x_h[:, None] + delta_xh[None, :] >= 0) & (off_x_h[:, None] +
            delta_xh[None, :] < IN_H) & (off_x_w[:, None] + delta_xw[None,
            :] >= 0) & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
    acc = acc
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w
    y_ptrs = y + off_y_n[:, None] * stride_yn + off_y_h[:, None
        ] * stride_yh + off_y_w[:, None] * stride_yw + off_y_k[None, :
        ] * stride_yc
    mask_y = (off_y_n < BATCH)[:, None] & (off_y_h < OUT_H + output_padding_h)[
        :, None] & (off_y_w < OUT_W + output_padding_w)[:, None] & (off_y_k <
        KERNEL_N)[None, :]
    tl.store(y_ptrs, acc, mask=mask_y)
    return
