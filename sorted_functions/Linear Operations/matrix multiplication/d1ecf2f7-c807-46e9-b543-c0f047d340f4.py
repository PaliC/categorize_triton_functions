import triton
import triton.language as tl
import torch

@triton_autotune(configs=_add_embeddings_bwd_configs(), key=[
    'AUTOTUNE_MAX_SEQ_LEN', 'AUTOTUNE_B', 'D'])
@triton.jit
def _add_embeddings_bwd_kernel(In, KeyInds, ValueInds, Out,
    AUTOTUNE_MAX_SEQ_LEN, B, AUTOTUNE_B, D, jagged_size, stride_in,
    stride_on, BLOCK_D: 'tl.constexpr', BLOCK: 'tl.constexpr'):
    off_block = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    key_ind = -1
    key_ind = key_ind
    accumulator = tl.zeros((BLOCK_D,), dtype=In.dtype.element_ty)
    for off_i in range(0, BLOCK):
        off = off_block * BLOCK + off_i
        if off < jagged_size:
            value_ind = tl.load(ValueInds + off)
            in_offset = In + value_ind * stride_in
            jagged_in = tl.load(in_offset + offs_d, mask=mask_d)
            key_ind_new = tl.load(KeyInds + off)
            if key_ind == key_ind_new:
                accumulator += jagged_in
            else:
                if key_ind >= 0:
                    out_offset = Out + key_ind * stride_on
                    tl.atomic_add(out_offset + offs_d, accumulator, mask=
                        mask_d, sem='relaxed')
                key_ind = key_ind_new
                accumulator = jagged_in
    if key_ind >= 0:
        out_offset = Out + key_ind * stride_on
        tl.atomic_add(out_offset + offs_d, accumulator, mask=mask_d, sem=
            'relaxed')
