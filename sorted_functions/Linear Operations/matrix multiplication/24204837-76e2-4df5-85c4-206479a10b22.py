import triton
import triton.language as tl
import torch

@triton.jit
def _contiguous_block(input_tiles, next_id, pid_n, input, other, output, K,
    N, stride_input_m, stride_input_k, stride_other_b, stride_other_k,
    stride_other_n, stride_output_m, stride_output_n, out_dtype:
    'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', TILE_M: 'tl.constexpr',
    TILE_N: 'tl.constexpr', TILE_K: 'tl.constexpr', EVEN_K: 'tl.constexpr',
    EVEN_N: 'tl.constexpr', EQUAL_K: 'tl.constexpr'):
    start_off = tl.load(input_tiles + 5 * next_id + 2)
    type_id = tl.load(input_tiles + 5 * next_id + 1)
    if EQUAL_K:
        _reg_matmul(pid_n, type_id, start_off, input, other, output, N,
            stride_input_m, stride_input_k, stride_other_b, stride_other_k,
            stride_other_n, stride_output_m, stride_output_n, out_dtype=
            out_dtype, BLOCK_SIZE=BLOCK_SIZE, TILE_M=TILE_M, TILE_N=TILE_N,
            TILE_K=TILE_K, EVEN_N=EVEN_N)
    else:
        for i in range(0, BLOCK_SIZE):
            _general_matmul(pid_n, start_off + i * TILE_M, start_off + (i +
                1) * TILE_M, input, other + type_id * stride_other_b,
                output, K, N, stride_input_m, stride_input_k,
                stride_other_k, stride_other_n, stride_output_m,
                stride_output_n, out_dtype=out_dtype, MASK_M=True, EVEN_K=
                EVEN_K, EVEN_N=EVEN_N, TILE_M=TILE_M, TILE_N=TILE_N, TILE_K
                =TILE_K)
