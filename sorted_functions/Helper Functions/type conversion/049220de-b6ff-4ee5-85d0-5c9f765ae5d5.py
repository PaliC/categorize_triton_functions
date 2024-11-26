import triton
import triton.language as tl
import torch

@triton.jit
def load_rotary_embedded_vector(QK, stride_qk_n, stride_qk_t, stride_qk_hid,
    COS, stride_cos_t, stride_cos_hid, SIN, stride_sin_t, stride_sin_hid,
    idx_n, idx_t_qk, idx_t_rope, HID, BLOCK_HID):
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    idx_hid_rot = (idx_hid + HID // 2) % HID
    mask_hid_rot = mask_hid
    vec = tl.load(QK + idx_n * stride_qk_n + idx_t_qk * stride_qk_t + 
        idx_hid * stride_qk_hid, mask=mask_hid, other=0)
    vec_rot = tl.load(QK + idx_n * stride_qk_n + idx_t_qk * stride_qk_t + 
        idx_hid_rot * stride_qk_hid, mask=mask_hid_rot, other=0)
    vec_rot = tl.where(idx_hid < HID // 2, -vec_rot, vec_rot)
    cos = tl.load(COS + idx_t_rope * stride_cos_t + idx_hid *
        stride_cos_hid, mask=mask_hid, other=0)
    sin = tl.load(SIN + idx_t_rope * stride_sin_t + idx_hid *
        stride_sin_hid, mask=mask_hid, other=0)
    vec_rope = vec.to(tl.float32) * cos + vec_rot.to(tl.float32) * sin
    return vec_rope, vec, vec_rot, cos, sin
