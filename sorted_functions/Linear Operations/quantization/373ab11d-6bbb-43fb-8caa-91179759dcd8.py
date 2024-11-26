import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_K': 16}, num_stages=2,
    num_warps=4), triton.Config({'BLOCK_SIZE_K': 32}, num_stages=2,
    num_warps=4), triton.Config({'BLOCK_SIZE_K': 64}, num_stages=2,
    num_warps=4), triton.Config({'BLOCK_SIZE_K': 16}, num_stages=2,
    num_warps=4), triton.Config({'BLOCK_SIZE_K': 32}, num_stages=2,
    num_warps=2), triton.Config({'BLOCK_SIZE_K': 16}, num_stages=3,
    num_warps=4), triton.Config({'BLOCK_SIZE_K': 32}, num_stages=3,
    num_warps=2), triton.Config({'BLOCK_SIZE_K': 16}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_K': 32}, num_stages=4,
    num_warps=2), triton.Config({'BLOCK_SIZE_K': 128}, num_stages=2,
    num_warps=2), triton.Config({'BLOCK_SIZE_K': 128}, num_stages=1,
    num_warps=4)], key=['B', 'M', 'N'])
@triton.jit
def matmul_quant_kernel(b_ptr, c_ptr, res_ptr, output_scale, B, M:
    'tl.constexpr', N: 'tl.constexpr', np2_M: 'tl.constexpr', np2_N:
    'tl.constexpr', stride_bb, stride_bk, stride_bn, stride_ck, stride_cn,
    stride_resb, stride_resm, stride_resn, BLOCK_SIZE_K: 'tl.constexpr'):
    """
    Quant(b @ c)

    b [B, M, N]
    c [N, N]
    """
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1) + tl.program_id(axis=2) * tl.num_programs(
        axis=1)
    pid_m = pid
    offs_bm = (pid_m * M + tl.arange(0, np2_M)) % M
    offs_cn = tl.arange(0, np2_N) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = b_ptr + batch_id * stride_bb + (offs_bm[:, None] * stride_bk +
        offs_k[None, :] * stride_bn)
    c_ptrs = c_ptr + (offs_k[:, None] * stride_ck + offs_cn[None, :] *
        stride_cn)
    accumulator = tl.zeros((np2_M, np2_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        b = tl.load(b_ptrs, mask=offs_k[None, :] < N - k * BLOCK_SIZE_K,
            other=0.0)
        c = tl.load(c_ptrs, mask=offs_k[:, None] < N - k * BLOCK_SIZE_K,
            other=0.0)
        accumulator = tl.dot(b, c, accumulator)
        b_ptrs += BLOCK_SIZE_K * stride_bn
        c_ptrs += BLOCK_SIZE_K * stride_ck
    abs_src_val = tl.abs(accumulator)
    max_src_val = tl.max(abs_src_val)
    scale = max_src_val / 7.0
    quant_val = libdevice.llrint(accumulator / scale)
    quant_val = max(-8, min(quant_val, 7))
    quant_val = quant_val.reshape(np2_M, np2_N // 2, 2, can_reorder=False)
    quant_val_even, quant_val_odd = quant_val.split()
    quant_val_odd = quant_val_odd << 4
    res = tl.zeros((np2_M, np2_N // 2), dtype=tl.int8)
    res = res | quant_val_odd & 240
    res = res | quant_val_even & 15
    offs_resm = pid_m * M + tl.arange(0, np2_M)
    offs_resn = tl.arange(0, np2_N // 2)
    res_ptrs = res_ptr + stride_resb * batch_id + stride_resm * offs_resm[:,
        None] + stride_resn * offs_resn[None, :]
    res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N // 2)
    tl.store(res_ptrs, res, mask=res_mask)
    tl.store(output_scale + batch_id, scale)
