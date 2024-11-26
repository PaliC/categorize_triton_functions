import triton
import triton.language as tl
import torch

@triton.jit
def quant_dot_kernel(scale_ptr, offset_ptr, weight_ptr, activation_ptr,
    z_ptr, N0, N1, MID, B0: 'tl.constexpr', B1: 'tl.constexpr', B_MID:
    'tl.constexpr'):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    off_j = block_id_j * B0 + tl.arange(0, B0)
    off_k = block_id_k * B1 + tl.arange(0, B1)
    mask_j = off_j < N0
    mask_k = off_k < N1
    z = tl.zeros((B0, B1), dtype=tl.float32)
    off_z = off_j[:, None] * N1 + off_k[None, :]
    mask_z = mask_j[:, None] & mask_k[None, :]
    for l in tl.range(0, MID, B_MID):
        off_l_div_g = tl.arange(0, B_MID // GROUP) + l // GROUP
        mask_l_div_g = off_l_div_g < MID // GROUP
        off_scale = off_j[:, None] * (MID // GROUP) + off_l_div_g[None, :]
        mask_scale = mask_j[:, None] & mask_l_div_g[None, :]
        scale = tl.load(scale_ptr + off_scale, mask=mask_scale)
        shift = tl.load(offset_ptr + off_j, mask=mask_j)
        off_weight_l = l + tl.arange(0, B_MID // FPINT)
        mask_weight_l = off_weight_l < MID // FPINT
        off_weight = off_j[:, None] * (MID // FPINT) + off_weight_l[None, :]
        mask_weight = mask_j[:, None] & mask_weight_l[None, :]
        weight = tl.load(weight_ptr + off_weight, mask=mask_weight)
        off_l = l + tl.arange(0, B_MID)
        mask_l = off_l < MID
        off_activation = off_l[:, None] * N1 + off_k[None, :]
        mask_activation = mask_l[:, None] & mask_k[None, :]
        activation = tl.load(activation_ptr + off_activation, mask=
            mask_activation)
        BITS = 32 // FPINT
        unpack_offs = tl.arange(0, FPINT) * BITS
        unpack_upperbound_mask = (1 << BITS) - 1
        unpacked_shift = shift[:, None] >> unpack_offs & unpack_upperbound_mask
        unpacked_weight = weight[:, :, None
            ] >> unpack_offs & unpack_upperbound_mask
        transformed_weight = scale[:, :, None] * (unpacked_weight -
            unpacked_shift[:, :, None])
        transformed_weight = transformed_weight.reshape(unpacked_shift.
            shape[0], unpacked_shift.shape[-1] * FPINT)
        z += tl.dot(transformed_weight, activation)
    tl.store(z_ptr + off_z, z, mask=mask_z)
    return
