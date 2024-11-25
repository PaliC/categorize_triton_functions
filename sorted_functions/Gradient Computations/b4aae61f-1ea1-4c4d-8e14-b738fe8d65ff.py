import triton
import triton.language as tl
import torch

@triton.jit
def _bwd_do_attn_kernel(O, Do, De, stride_ob: 'int', stride_om: 'int',
    stride_oh: 'int', stride_dob: 'int', stride_dom: 'int', stride_doh:
    'int', stride_deb: 'int', stride_deh: 'int', nheads: 'int', headdim:
    'int', seqlen_q: 'int', BLOCK_M: 'tl.constexpr', BLOCK_HEADDIM:
    'tl.constexpr'):
    """Triton kernel for the backward pass of the attention mechanism with respect to the output gradient.

	Args:
		O: Output array.
		Do: Output gradient array.
		De: Delta array.
		stride_ob: Stride for the output batch dimension.
		stride_om: Stride for the output sequence dimension.
		stride_oh: Stride for the output head dimension.
		stride_dob: Stride for the output gradient batch dimension.
		stride_dom: Stride for the output gradient sequence dimension.
		stride_doh: Stride for the output gradient head dimension.
		stride_deb: Stride for the delta batch dimension.
		stride_deh: Stride for the delta head dimension.
		nheads: Number of heads.
		headdim: Head dimension.
		seqlen_q: Sequence length of the query.
		BLOCK_M: Block size for the query sequence dimension.
		BLOCK_HEADDIM: Block size for the head dimension.
	"""
    off_q = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // nheads
    off_h = off_bh % nheads
    offs_m = off_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None
        ] * stride_om + offs_d[None, :]
    do_ptrs = Do + off_b * stride_dob + off_h * stride_doh + offs_m[:, None
        ] * stride_dom + offs_d[None, :]
    o = tl.load(o_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :
        ] < headdim), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None,
        :] < headdim), other=0.0)
    delta = tl.sum(o * do, axis=1)
    tl.store(De + (off_b * stride_deb + off_h * stride_deh + offs_m), delta,
        mask=offs_m < seqlen_q)
