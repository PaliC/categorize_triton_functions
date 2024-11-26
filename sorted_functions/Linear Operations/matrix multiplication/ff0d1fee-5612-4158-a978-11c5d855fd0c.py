import triton
import triton.language as tl
import torch

@triton.autotune(configs=_generate_configs(), reset_to_zero=['grad_other'],
    key=['N', 'K', 'stddev_tile_size_m', 'avg_tile_size_m'],
    prune_configs_by={'early_config_prune': functools.partial(
    _early_config_prune, is_weight=True), 'perf_model': _weight_perf_model,
    'top_k': 100 if GlobalConfig.with_perf_model else 10}, use_cuda_graph=
    True, rep=20)
@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['TILE_SIZE_K'] ==
    0, 'EVEN_N': lambda args: args['N'] % args['TILE_SIZE_N'] == 0})
@triton.jit
def split_matmul_kernel(input, input_slices, input_tiles, grad_output,
    grad_other, grad_other_tiles, K, N, stride_input_m, stride_input_k,
    stride_grad_output_m, stride_grad_output_n, stride_grad_other_b,
    stride_grad_other_k, stride_grad_other_n, stddev_tile_size_m,
    avg_tile_size_m, out_dtype: 'tl.constexpr', NUM_BLOCKS: 'tl.constexpr',
    NUM_TILES: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', TILE_SIZE_M:
    'tl.constexpr', TILE_SIZE_N: 'tl.constexpr', TILE_SIZE_K:
    'tl.constexpr', EVEN_K: 'tl.constexpr', EVEN_N: 'tl.constexpr',
    DETERMINISTIC: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    grid_k = tl.cdiv(K, TILE_SIZE_K)
    grid_n = tl.cdiv(N, TILE_SIZE_N)
    next_id = pid // (grid_k * grid_n)
    next_next_id = tl.load(input_tiles + 5 * next_id + 4)
    tile_id = pid % (grid_k * grid_n)
    pid_k = tile_id // grid_n
    pid_n = tile_id % grid_n
    if next_next_id == 0:
        slice_id = tl.load(input_tiles + 5 * next_id + 0)
        type_id = tl.load(input_tiles + 5 * next_id + 1)
        start_off = tl.load(input_tiles + 5 * next_id + 2)
        slice_start = tl.load(input_slices + 5 * slice_id + 2)
        slice_end = tl.load(input_slices + 5 * slice_id + 3)
        M = slice_end - slice_start
        _dynamic_matmul(pid_k, pid_n, next_id, input + start_off *
            stride_input_m, grad_output + start_off * stride_grad_output_m,
            grad_other + type_id * stride_grad_other_b, grad_other_tiles,
            stride_input_m, stride_input_k, stride_grad_output_m,
            stride_grad_output_n, stride_grad_other_b, stride_grad_other_k,
            stride_grad_other_n, K, N, M, TILE_SIZE_M * BLOCK_SIZE,
            out_dtype=out_dtype, BLOCK_LENGTH=TILE_SIZE_M * BLOCK_SIZE,
            TILE_K=TILE_SIZE_K, TILE_N=TILE_SIZE_N, TILE_M=TILE_SIZE_M,
            EVEN_K=EVEN_K, EVEN_N=EVEN_N, EVEN_M=True, DETERMINISTIC=
            DETERMINISTIC)
    else:
        _split_noncontiguous_block(pid_k, pid_n, input, input_slices,
            input_tiles, grad_output, grad_other, grad_other_tiles,
            stride_input_m, stride_input_k, stride_grad_output_m,
            stride_grad_output_n, stride_grad_other_b, stride_grad_other_k,
            stride_grad_other_n, K, N, next_id, next_next_id, out_dtype=
            out_dtype, BLOCK_SIZE=BLOCK_SIZE, NUM_TILES=NUM_TILES, TILE_K=
            TILE_SIZE_K, TILE_N=TILE_SIZE_N, TILE_M=TILE_SIZE_M, EVEN_K=
            EVEN_K, EVEN_N=EVEN_N, DETERMINISTIC=DETERMINISTIC)
