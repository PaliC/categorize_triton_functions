import triton
import triton.language as tl
import torch

@triton.autotune(configs=_generate_configs(), key=['N', 'K',
    'stddev_tile_size_m', 'avg_tile_size_m'], prune_configs_by={
    'early_config_prune': functools.partial(_early_config_prune, is_weight=
    False)}, rep=10, use_cuda_graph=True)
@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['TILE_SIZE_K'] ==
    0, 'EVEN_N': lambda args: args['N'] % args['TILE_SIZE_N'] == 0,
    'EQUAL_K': lambda args: args['K'] == args['TILE_SIZE_K']})
@triton.jit
def segment_matmul_kernel(input, input_tiles, other, output, K, N,
    stride_input_m, stride_input_k, stride_other_b, stride_other_k,
    stride_other_n, stride_output_m, stride_output_n, stddev_tile_size_m,
    avg_tile_size_m, out_dtype: 'tl.constexpr', NUM_TILES: 'tl.constexpr',
    NUM_BLOCKS: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', TILE_SIZE_M:
    'tl.constexpr', EVEN_K: 'tl.constexpr', EVEN_N: 'tl.constexpr', EQUAL_K:
    'tl.constexpr', TILE_SIZE_N: 'tl.constexpr', TILE_SIZE_K: 'tl.constexpr'):
    TILE_N: 'tl.constexpr' = TILE_SIZE_N
    TILE_K: 'tl.constexpr' = TILE_SIZE_K
    TILE_M: 'tl.constexpr' = TILE_SIZE_M
    GROUP_M: 'tl.constexpr' = 4
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, TILE_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(NUM_BLOCKS - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    next_id = pid_m
    next_next_id = tl.load(input_tiles + 5 * next_id + 4)
    if next_next_id == 0:
        _contiguous_block(input_tiles, next_id, pid_n, input, other, output,
            K, N, stride_input_m, stride_input_k, stride_other_b,
            stride_other_k, stride_other_n, stride_output_m,
            stride_output_n, out_dtype=out_dtype, BLOCK_SIZE=BLOCK_SIZE,
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K, EVEN_K=EVEN_K,
            EVEN_N=EVEN_N, EQUAL_K=EQUAL_K)
    else:
        _noncontiguous_block(input_tiles, next_id, next_next_id, pid_n,
            input, other, output, K, N, stride_input_m, stride_input_k,
            stride_other_b, stride_other_k, stride_other_n, stride_output_m,
            stride_output_n, out_dtype=out_dtype, BLOCK_SIZE=BLOCK_SIZE,
            NUM_TILES=NUM_TILES, TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=
            TILE_K, EVEN_K=EVEN_K, EVEN_N=EVEN_N)
