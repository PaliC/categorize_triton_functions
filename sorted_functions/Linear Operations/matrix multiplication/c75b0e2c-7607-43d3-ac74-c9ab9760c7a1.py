import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({}, num_stages=2, num_warps=4),
    triton.Config({}, num_stages=2, num_warps=2), triton.Config({},
    num_stages=3, num_warps=4), triton.Config({}, num_stages=3, num_warps=2
    ), triton.Config({}, num_stages=4, num_warps=4), triton.Config({},
    num_stages=4, num_warps=2)], key=['B', 'M', 'N'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, res_ptr, output_scale, B, M:
    'tl.constexpr', N: 'tl.constexpr', np2_M: 'tl.constexpr', np2_N:
    'tl.constexpr', stride_am, stride_ak, stride_bb, stride_bk, stride_bn,
    stride_ck, stride_cn, stride_resb, stride_resm, stride_resn,
    BLOCK_SIZE_M: 'tl.constexpr', is_split: 'tl.constexpr'):
    """
    a @ b @ c

    a [M, M]
    b [B, M, N]
    c [N, N]

    now only supports BLOCK_SIZE_M == triton.next_power_of_2(BLOCK_SIZE_M)
    """
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1) + tl.program_id(axis=2) * tl.num_programs(
        axis=1)
    pid_m = pid
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = tl.arange(0, np2_N) % N
    offs_k = tl.arange(0, np2_M)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
        stride_ak)
    b_ptrs = b_ptr + batch_id * stride_bb + (offs_k[:, None] * stride_bk + 
        offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, np2_N), dtype=tl.float32)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < M, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < M, other=0.0)
    accumulator += tl.dot(a, b)
    tmp_ab = accumulator
    offs_cn = tl.arange(0, np2_N) % N
    offs_k = tl.arange(0, np2_N)
    c_ptrs = c_ptr + (offs_k[:, None] * stride_ck + offs_cn[None, :] *
        stride_cn)
    c = tl.load(c_ptrs, mask=offs_k[:, None] < N, other=0.0)
    accumulator = 0
    accumulator += tl.dot(tmp_ab, c)
    if is_split:
        res = accumulator
        offs_resm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_resn = tl.arange(0, np2_N)
        res_ptrs = res_ptr + stride_resb * batch_id + stride_resm * offs_resm[
            :, None] + stride_resn * offs_resn[None, :]
        res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N)
        tl.store(res_ptrs, res, mask=res_mask)
    else:
        abs_src_val = tl.abs(accumulator)
        max_src_val = tl.max(abs_src_val)
        scale = max_src_val / 7.0
        quant_val = libdevice.llrint(accumulator / scale)
        quant_val = max(-8, min(quant_val, 7))
        quant_val = quant_val.reshape(BLOCK_SIZE_M, np2_N // 2, 2,
            can_reorder=False)
        quant_val_even, quant_val_odd = quant_val.split()
        quant_val_odd = quant_val_odd << 4
        res = tl.zeros((BLOCK_SIZE_M, np2_N // 2), dtype=tl.int8)
        res = res | quant_val_odd & 240
        res = res | quant_val_even & 15
        offs_resm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_resn = tl.arange(0, np2_N // 2)
        res_ptrs = res_ptr + stride_resb * batch_id + stride_resm * offs_resm[
            :, None] + stride_resn * offs_resn[None, :]
        res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N // 2)
        tl.store(res_ptrs, res, mask=res_mask)
        tl.store(output_scale + batch_id, scale)
