import triton
import triton.language as tl
import torch

@triton.jit
def grad_rotary_embedded_vector(grad_vec_rope, vec_origin, vec_rot, cos,
    sin, HID, BLOCK_HID):
    grad_vec_origin = grad_vec_rope * cos
    idx_vec_origin_hid = tl.arange(0, BLOCK_HID)
    grad_vec_rot = grad_vec_rope * sin
    grad_vec_rot = tl.where(idx_vec_origin_hid < HID // 2, -grad_vec_rot,
        grad_vec_rot)
    idx_vec_rot_hid = (idx_vec_origin_hid + HID // 2) % HID
    return grad_vec_origin, idx_vec_origin_hid, grad_vec_rot, idx_vec_rot_hid
