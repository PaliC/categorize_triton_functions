import triton
import triton.language as tl
import torch

@triton.jit
def _split_noncontiguous_block(pid_k, pid_n, input, input_slices,
    input_tiles, grad_output, grad_other, grad_other_tiles, stride_input_m,
    stride_input_k, stride_grad_output_m, stride_grad_output_n,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n, K, N,
    next_id, next_next_id, out_dtype: 'tl.constexpr', BLOCK_SIZE:
    'tl.constexpr', NUM_TILES: 'tl.constexpr', TILE_K: 'tl.constexpr',
    TILE_N: 'tl.constexpr', TILE_M: 'tl.constexpr', EVEN_K: 'tl.constexpr',
    EVEN_N: 'tl.constexpr', DETERMINISTIC: 'tl.constexpr'):
    for _ in range(0, BLOCK_SIZE):
        if next_id < NUM_TILES and next_id != -1:
            start_off = tl.load(input_tiles + 5 * next_id + 2)
            end_off = tl.load(input_tiles + 5 * next_id + 3)
            length = end_off - start_off
            if length > 0:
                type_id = tl.load(input_tiles + 5 * next_id + 1)
                slice_id = tl.load(input_tiles + 5 * next_id + 0)
                slice_start = tl.load(input_slices + 5 * slice_id + 2)
                slice_end = tl.load(input_slices + 5 * slice_id + 3)
                M = slice_end - slice_start
                _split_dispatch(pid_k, pid_n, next_id, input + start_off *
                    stride_input_m, grad_output + start_off *
                    stride_grad_output_m, grad_other + type_id *
                    stride_grad_other_b, grad_other_tiles, stride_input_m,
                    stride_input_k, stride_grad_output_m,
                    stride_grad_output_n, stride_grad_other_b,
                    stride_grad_other_k, stride_grad_other_n, K, N, M,
                    length, out_dtype=out_dtype, BLOCK_LENGTH=TILE_M *
                    BLOCK_SIZE, TILE_K=TILE_K, TILE_N=TILE_N, TILE_M=TILE_M,
                    EVEN_K=EVEN_K, EVEN_N=EVEN_N, EVEN_M=False,
                    DYNAMIC_TILING=True, DETERMINISTIC=DETERMINISTIC)
            next_id = next_next_id
            next_next_id += 1
