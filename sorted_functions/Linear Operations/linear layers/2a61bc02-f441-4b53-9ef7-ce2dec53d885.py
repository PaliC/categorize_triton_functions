import triton
import triton.language as tl
import torch

@triton.jit
def fused_embeddings_kernel(x_ptr, wte_ptr, wpe_ptr, z_ptr, B, L, V, P, H,
    dropout_prob=0.0, seed=1337, BLOCK_SIZE: 'tl.constexpr'=512):
    pid = tl.program_id(0)
    wte_ptr += tl.load(x_ptr + pid) * H
    wpe_ptr += pid % L * H
    z_ptr += pid * H
    for k in range(0, H, BLOCK_SIZE):
        offset = k + tl.arange(0, BLOCK_SIZE)
        mask = offset < H
        z = tl.load(wte_ptr + offset, mask=mask, other=0.0)
        z += tl.load(wpe_ptr + offset, mask=mask, other=0.0)
        z = dropout(z, dropout_prob, seed, offset)
        tl.store(z_ptr + offset, z, mask=mask)
