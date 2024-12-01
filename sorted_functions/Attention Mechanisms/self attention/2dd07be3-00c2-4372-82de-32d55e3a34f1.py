import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gsa_fwd_kernel_intra_V(q, k, g, A, scale, T: 'tl.constexpr', K:
    'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK:
    'tl.constexpr', NC: 'tl.constexpr', NK: 'tl.constexpr', NG: 'tl.constexpr'
    ):
    i_c, i_bh = tl.program_id(0), tl.program_id(1)
    for i_k in range(0, NK):
        chunk_gsa_fwd_kernel_intra_Vk(q, k, g, A, i_k, i_c, i_bh, scale, T=
            T, K=K, BT=BT, BC=BC, BK=BK, NC=NC, NG=NG)
