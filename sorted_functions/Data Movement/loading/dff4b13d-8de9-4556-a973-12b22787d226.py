import triton
import triton.language as tl
import torch

@triton.jit
def _noncontiguous_block(input_tiles, next_id, next_next_id, pid_n, input,
    other, output, K, N, stride_input_m, stride_input_k, stride_other_b,
    stride_other_k, stride_other_n, stride_output_m, stride_output_n,
    out_dtype: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', NUM_TILES:
    'tl.constexpr', TILE_M: 'tl.constexpr', TILE_N: 'tl.constexpr', TILE_K:
    'tl.constexpr', EVEN_K: 'tl.constexpr', EVEN_N: 'tl.constexpr'):
    for _ in range(0, BLOCK_SIZE):
        if next_id < NUM_TILES and next_id != -1:
            start_off = tl.load(input_tiles + 5 * next_id + 2)
            end_off = tl.load(input_tiles + 5 * next_id + 3)
            length = end_off - start_off
            if length > 0:
                type_id = tl.load(input_tiles + 5 * next_id + 1)
                for i in range(0, tl.cdiv(length, TILE_M)):
                    _dispatch(pid_n, start_off + i * TILE_M, end_off, input,
                        other + type_id * stride_other_b, output, K, N,
                        stride_input_m, stride_input_k, stride_other_k,
                        stride_other_n, stride_output_m, stride_output_n,
                        out_dtype=out_dtype, MASK_M=True, EVEN_K=EVEN_K,
                        EVEN_N=EVEN_N, TILE_M=TILE_M, TILE_N=TILE_N, TILE_K
                        =TILE_K, DYNAMIC_TILING=True)
            next_id = next_next_id
            next_next_id += 1
