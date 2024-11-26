import triton
import triton.language as tl
import torch

@triton.jit
def _attention_scores_compute(Q, stride_q_n, stride_q_tdst, stride_q_hid, K,
    stride_k_n, stride_k_tsrc, stride_k_hid, COS, stride_cos_t,
    stride_cos_hid, SIN, stride_sin_t, stride_sin_hid, INDICES,
    stride_indices_d, stride_indices_z, VALUES, stride_values_z, N, TDST,
    TSRC, HID, NUM_SINK, WINDOW_SIZE, BLOCK_HID: 'tl.constexpr'):
    idx_n = tl.program_id(0)
    idx_tdst = tl.program_id(1)
    idx_k = tl.program_id(2)
    tdst = idx_tdst + TSRC - TDST
    if idx_k < NUM_SINK:
        idx_tsrc = idx_k
    else:
        window_offset = idx_k - NUM_SINK
        t_tsrc = tdst - WINDOW_SIZE + 1 + window_offset
        idx_tsrc = tl.maximum(idx_k, t_tsrc)
    key, _, _, _, _ = load_rotary_embedded_vector(K, stride_k_n,
        stride_k_tsrc, stride_k_hid, COS, stride_cos_t, stride_cos_hid, SIN,
        stride_sin_t, stride_sin_hid, idx_n, idx_tsrc, idx_k, HID, BLOCK_HID)
    query, _, _, _, _ = load_rotary_embedded_vector(Q, stride_q_n,
        stride_q_tdst, stride_q_hid, COS, stride_cos_t, stride_cos_hid, SIN,
        stride_sin_t, stride_sin_hid, idx_n, idx_tdst, tl.minimum(tdst, 
        WINDOW_SIZE + NUM_SINK - 1), HID, BLOCK_HID)
    score = tl.sum(query * key)
    score = score * (1 / tl.sqrt(HID))
    score = tl.where(idx_tsrc <= tdst, score, float('-inf'))
    idx_z = idx_n * TDST * (WINDOW_SIZE + NUM_SINK) + idx_tdst * (WINDOW_SIZE +
        NUM_SINK) + idx_k
    tl.store(VALUES + idx_z * stride_values_z, value=score)
    tl.store(INDICES + 0 * stride_indices_d + idx_z * stride_indices_z,
        value=idx_n)
    tl.store(INDICES + 1 * stride_indices_d + idx_z * stride_indices_z,
        value=idx_tdst)
    tl.store(INDICES + 2 * stride_indices_d + idx_z * stride_indices_z,
        value=idx_tsrc)
