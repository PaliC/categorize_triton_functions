import triton
import triton.language as tl
import torch

@triton.jit
def _dynamic_matmul(pid_k, pid_n, next_id, input, grad_output, grad_other,
    grad_other_tiles, stride_input_m, stride_input_k, stride_grad_output_m,
    stride_grad_output_n, stride_grad_other_b, stride_grad_other_k,
    stride_grad_other_n, K, N, M, length, out_dtype: 'tl.constexpr',
    BLOCK_LENGTH: 'tl.constexpr', TILE_K: 'tl.constexpr', TILE_N:
    'tl.constexpr', TILE_M: 'tl.constexpr', EVEN_N: 'tl.constexpr', EVEN_K:
    'tl.constexpr', EVEN_M: 'tl.constexpr', DETERMINISTIC: 'tl.constexpr'):
    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_m = tl.arange(0, TILE_M)
    acc = tl.zeros((TILE_K, TILE_N), dtype=out_dtype)
    mask_k = offs_k[:, None] < K if not EVEN_K else True
    mask_n = offs_n[None, :] < N if not EVEN_N else True
    input_ptrs = input + (offs_m[None, :] * stride_input_m + offs_k[:, None
        ] * stride_input_k)
    grad_output_ptrs = grad_output + (offs_m[:, None] *
        stride_grad_output_m + offs_n[None, :] * stride_grad_output_n)
    m_iter = length // TILE_M if EVEN_M else tl.cdiv(length, TILE_M)
    for m in range(0, m_iter):
        if EVEN_K:
            if EVEN_M:
                a = tl.load(input_ptrs)
            else:
                a = tl.load(input_ptrs, mask=offs_m[None, :] + m * TILE_M <
                    length, other=0.0)
        elif EVEN_M:
            a = tl.load(input_ptrs, mask=mask_k, other=0.0)
        else:
            a = tl.load(input_ptrs, mask=mask_k & (offs_m[None, :] + m *
                TILE_M < length), other=0.0)
        if EVEN_N:
            if EVEN_M:
                b = tl.load(grad_output_ptrs)
            else:
                b = tl.load(grad_output_ptrs, mask=offs_m[:, None] + m *
                    TILE_M < length, other=0.0)
        elif EVEN_M:
            b = tl.load(grad_output_ptrs, mask=mask_n)
        else:
            b = tl.load(grad_output_ptrs, mask=mask_n & (offs_m[:, None] + 
                m * TILE_M < length), other=0.0)
        acc += tl.dot(a, b, out_dtype=out_dtype)
        input_ptrs += TILE_M * stride_input_m
        grad_output_ptrs += TILE_M * stride_grad_output_m
    acc = acc
    if DETERMINISTIC:
        if M <= BLOCK_LENGTH:
            c_ptrs = grad_other + stride_grad_other_k * offs_k[:, None
                ] + stride_grad_other_n * offs_n[None, :]
            if EVEN_N and EVEN_K:
                tl.store(c_ptrs, acc)
            else:
                c_mask = mask_k & mask_n
                tl.store(c_ptrs, acc, mask=c_mask)
        else:
            c_ptrs = (grad_other_tiles + next_id * stride_grad_other_b + 
                stride_grad_other_k * offs_k[:, None] + stride_grad_other_n *
                offs_n[None, :])
            if EVEN_N and EVEN_K:
                tl.store(c_ptrs, acc)
            else:
                c_mask = mask_k & mask_n
                tl.store(c_ptrs, acc, mask=c_mask)
    else:
        c_ptrs = grad_other + stride_grad_other_k * offs_k[:, None
            ] + stride_grad_other_n * offs_n[None, :]
        if M <= BLOCK_LENGTH:
            if EVEN_N and EVEN_K:
                tl.store(c_ptrs, acc)
            else:
                c_mask = mask_k & mask_n
                tl.store(c_ptrs, acc, mask=c_mask)
        elif EVEN_N and EVEN_K:
            tl.atomic_add(c_ptrs, acc)
        else:
            c_mask = mask_k & mask_n
            tl.atomic_add(c_ptrs, acc, mask=c_mask)
