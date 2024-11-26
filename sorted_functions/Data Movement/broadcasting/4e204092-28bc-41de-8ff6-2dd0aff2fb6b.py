import triton
import triton.language as tl
import torch

@triton.autotune(configs=_generate_reduce_configs(), key=['N', 'K'], rep=1,
    use_cuda_graph=True)
@triton.jit
def split_reduce_kernel(slice_to_tiles, grad_other_tiles, grad_other,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n, K, N,
    TILE_SIZE_N: 'tl.constexpr', TILE_SIZE_K: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    grid_k = tl.cdiv(K, TILE_SIZE_K)
    grid_n = tl.cdiv(N, TILE_SIZE_N)
    slice_id = pid // (grid_k * grid_n)
    pid_k = pid % (grid_k * grid_n) // grid_n
    pid_n = pid % (grid_k * grid_n) % grid_n
    type_id = tl.load(slice_to_tiles + slice_id * 3 + 0)
    start_tile_id = tl.load(slice_to_tiles + slice_id * 3 + 1)
    end_tile_id = tl.load(slice_to_tiles + slice_id * 3 + 2)
    if start_tile_id == end_tile_id or end_tile_id - start_tile_id == 1:
        return
    acc = tl.zeros((TILE_SIZE_K, TILE_SIZE_N), dtype=grad_other.dtype.
        element_ty)
    k_offs = pid_k * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[:, None]
    n_offs = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)[None, :]
    grad_other_tiles_ptrs = (grad_other_tiles + k_offs *
        stride_grad_other_k + n_offs * stride_grad_other_n)
    mask = (k_offs < K) & (n_offs < N)
    for i in range(start_tile_id, end_tile_id):
        acc += tl.load(grad_other_tiles_ptrs + stride_grad_other_b * i,
            mask=mask)
    tl.store(grad_other + type_id * stride_grad_other_b + k_offs *
        stride_grad_other_k + n_offs * stride_grad_other_n, acc, mask=mask)
