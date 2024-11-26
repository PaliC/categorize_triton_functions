import triton
import triton.language as tl
import torch

@triton.jit
def _attention_score_backward_compute(GRAD_VALUES, stride_grad_values_z, Q,
    stride_q_n, stride_q_tdst, stride_q_hid, K, stride_k_n, stride_k_tsrc,
    stride_k_hid, INDICES, stride_indices_d, stride_indices_z, COS,
    stride_cos_t, stride_cos_hid, SIN, stride_sin_t, stride_sin_hid, GRAD_Q,
    stride_grad_q_n, stride_grad_q_tdst, stride_grad_q_hid, GRAD_K,
    stride_grad_k_n, stride_grad_k_tsrc, stride_grad_k_hid, N, TDST, TSRC,
    HID, NNZ, NUM_SINK, WINDOW_SIZE, BLOCK_HID: 'tl.constexpr'):
    idx_z = tl.program_id(0)
    idx_n = tl.load(INDICES + 0 * stride_indices_d + idx_z * stride_indices_z)
    idx_tdst = tl.load(INDICES + 1 * stride_indices_d + idx_z *
        stride_indices_z)
    idx_tsrc = tl.load(INDICES + 2 * stride_indices_d + idx_z *
        stride_indices_z)
    tdst = idx_tdst + TSRC - TDST
    idx_k = idx_z % (NUM_SINK + WINDOW_SIZE)
    key, key_origin, key_rot, cos_k, sin_k = load_rotary_embedded_vector(K,
        stride_k_n, stride_k_tsrc, stride_k_hid, COS, stride_cos_t,
        stride_cos_hid, SIN, stride_sin_t, stride_sin_hid, idx_n, idx_tsrc,
        idx_k, HID, BLOCK_HID)
    query, query_origin, query_rot, cos_q, sin_q = load_rotary_embedded_vector(
        Q, stride_q_n, stride_q_tdst, stride_q_hid, COS, stride_cos_t,
        stride_cos_hid, SIN, stride_sin_t, stride_sin_hid, idx_n, idx_tdst,
        tl.minimum(tdst, WINDOW_SIZE + NUM_SINK - 1), HID, BLOCK_HID)
    grad_score = tl.load(GRAD_VALUES + idx_z * stride_grad_values_z)
    grad_score = tl.where(idx_tsrc <= tdst, grad_score, 0)
    grad_score = grad_score * (1 / tl.sqrt(HID))
    grad_key = grad_score * query
    grad_query = grad_score * key
    grad_key_origin, idx_key_origin_hid, grad_key_rot, idx_key_rot_hid = (
        grad_rotary_embedded_vector(grad_key, key_origin, key_rot, cos_k,
        sin_k, HID, BLOCK_HID))
    (grad_query_origin, idx_query_origin_hid, grad_query_rot, idx_query_rot_hid
        ) = (grad_rotary_embedded_vector(grad_query, query_origin,
        query_rot, cos_q, sin_q, HID, BLOCK_HID))
    mask_hid = tl.arange(0, BLOCK_HID) < HID
    tl.atomic_add(GRAD_K + idx_n * stride_grad_k_n + idx_tsrc *
        stride_grad_k_tsrc + idx_key_origin_hid * stride_grad_k_hid, mask=
        mask_hid, val=grad_key_origin)
    tl.atomic_add(GRAD_K + idx_n * stride_grad_k_n + idx_tsrc *
        stride_grad_k_tsrc + idx_key_rot_hid * stride_grad_k_hid, mask=
        mask_hid, val=grad_key_rot)
    tl.atomic_add(GRAD_Q + idx_n * stride_grad_q_n + idx_tdst *
        stride_grad_q_tdst + idx_query_origin_hid * stride_grad_q_hid, mask
        =mask_hid, val=grad_query_origin)
    tl.atomic_add(GRAD_Q + idx_n * stride_grad_q_n + idx_tdst *
        stride_grad_q_tdst + idx_query_rot_hid * stride_grad_q_hid, mask=
        mask_hid, val=grad_query_rot)
