import triton
import triton.language as tl
import torch

@triton.jit
def _sparse_attention_compute(INDICES, stride_indices_d, stride_indices_z,
    VALUES, stride_values_z, V, stride_v_n, stride_v_tsrc, stride_v_hid,
    CONTEXT, stride_context_n, stride_context_tdst, stride_context_hid, N,
    TDST, TSRC, HID, BK, NUM_SINK, WINDOW_SIZE, BLOCK_HID: 'tl.constexpr',
    BLOCK_K: 'tl.constexpr'):
    idx_n = tl.program_id(0)
    idx_tdst = tl.program_id(1)
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    acc = tl.zeros((BLOCK_HID,), dtype=tl.float32)
    for idx_bk in range(BK):
        CACHE_SIZE = NUM_SINK + WINDOW_SIZE
        idx_k = idx_bk * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = idx_k < CACHE_SIZE
        idx_z = idx_n * TDST * CACHE_SIZE + idx_tdst * CACHE_SIZE + idx_k
        mask_z = mask_k
        idx_tsrc = tl.load(INDICES + 2 * stride_indices_d + idx_z *
            stride_indices_z, mask=mask_z, other=0)
        mask_tsrc = mask_z
        score = tl.load(VALUES + idx_z * stride_values_z, mask=mask_z, other=0)
        value = tl.load(V + idx_n * stride_v_n + idx_tsrc[:, None] *
            stride_v_tsrc + idx_hid[None, :] * stride_v_hid, mask=mask_tsrc
            [:, None] & mask_hid[None, :], other=0)
        context = tl.sum(score[:, None] * value, axis=0)
        acc += context
    tl.store(CONTEXT + idx_n * stride_context_n + idx_tdst *
        stride_context_tdst + idx_hid * stride_context_hid, mask=mask_hid,
        value=acc)
