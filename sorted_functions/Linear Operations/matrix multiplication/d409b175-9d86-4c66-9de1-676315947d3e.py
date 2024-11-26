import triton
import triton.language as tl
import torch

@triton.autotune(configs=_config_XtY(), key=['M', 'N', 'K'])
@triton.heuristics({'NO_K_MASK': lambda args: args['K'] % args['BLOCK_K'] ==
    0, 'NO_N_MASK': lambda args: args['N'] % args['BLOCK_N'] == 0})
@triton.jit
def _groupXtY(DY_ptr, stride_dym, stride_dyk, X_ptr, stride_xm, stride_xn,
    DW_ptr, stride_dwe, stride_dwk, stride_dwn, expert_offsets_ptr, M, K:
    'tl.constexpr', N: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', BLOCK_K: 'tl.constexpr', ACC_TYPE: 'tl.constexpr',
    allow_tf32: 'tl.constexpr', NO_K_MASK: 'tl.constexpr', NO_N_MASK:
    'tl.constexpr'):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)
    pid0, pid1 = tl.swizzle2d(pid0, pid1, num0, num1, 4)
    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1
    if E_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1)
    end_idx = tl.load(expert_offsets_ptr + E_idx)
    if end_idx > start_idx:
        M_block = tl.max_contiguous(start_idx + tl.arange(0, BLOCK_M), BLOCK_M)
        K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        K_mask = K_block < K
        K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K),
            BLOCK_K)
        N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        N_mask = N_block < N
        N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N),
            BLOCK_N)
        M_idxs = M_block
        xt_blk_ptrs = X_ptr + K_block[:, None] * stride_xn + M_idxs[None, :
            ] * stride_xm
        dy_blk_ptrs = DY_ptr + M_idxs[:, None] * stride_dym + N_block[None, :
            ] * stride_dyk
        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_TYPE)
        iters = tl.cdiv(end_idx - start_idx, BLOCK_M)
        for i in range(0, iters):
            M_mask = i * BLOCK_M + M_block < end_idx
            if NO_K_MASK:
                xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
            else:
                xt = tl.load(xt_blk_ptrs, mask=K_mask[:, None] & M_mask[
                    None, :])
            if NO_N_MASK:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
            else:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[
                    None, :])
            xt_blk_ptrs += BLOCK_M * stride_xm
            dy_blk_ptrs += BLOCK_M * stride_dym
            acc += tl.dot(xt, dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)
        DW_blk_ptrs = DW_ptr + E_idx * stride_dwe + K_block[:, None
            ] * stride_dwk + N_block[None, :] * stride_dwn
        acc = acc
        tl.store(DW_blk_ptrs, acc, mask=K_mask[:, None] & N_mask[None, :])
