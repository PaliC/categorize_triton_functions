import triton
import triton.language as tl
import torch

@triton.jit
def create_flashinfer_kv_indices_triton(req_to_token_ptr,
    req_pool_indices_ptr, page_kernel_lens_ptr, kv_indptr, kv_start_idx,
    kv_indices_ptr, max_context_len: 'tl.constexpr'):
    BLOCK_SIZE: 'tl.constexpr' = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)
    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid)
    req_to_token_ptr += req_pool_index * max_context_len
    kv_indices_ptr += kv_indices_offset
    ld_offset = kv_start + tl.arange(0, BLOCK_SIZE)
    st_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = ld_offset < kv_end
        data = tl.load(req_to_token_ptr + ld_offset, mask=mask)
        tl.store(kv_indices_ptr + st_offset, data, mask=mask)
        ld_offset += BLOCK_SIZE
        st_offset += BLOCK_SIZE
