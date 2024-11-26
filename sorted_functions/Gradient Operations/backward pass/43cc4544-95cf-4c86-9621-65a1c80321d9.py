import triton
import triton.language as tl
import torch

@triton.jit
def _add_position_embeddings_bwd_kernel(Jagged, seq_offsets, high_inds,
    DenseOut, JaggedOut, B, D, scale, stride_jn, stride_jon, stride_don,
    SCALE_JAGGED: 'tl.constexpr', BLOCK_D: 'tl.constexpr'):
    off_k = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for off_b in range(0, B):
        max_ind = tl.load(high_inds + off_b)
        if off_k < max_ind:
            seq_start = tl.load(seq_offsets + off_b)
            jagged_ptr = Jagged + seq_start * stride_jn + off_k * stride_jn
            jagged_ptrs = jagged_ptr + offs_d
            jg = tl.load(jagged_ptrs, mask=offs_d < D)
            accumulator += jg
            if SCALE_JAGGED:
                out_jagged_ptr = (JaggedOut + seq_start * stride_jon + 
                    off_k * stride_jon)
                out_jagged_ptrs = out_jagged_ptr + offs_d
                tl.store(out_jagged_ptrs, jg * scale, mask=offs_d < D)
        elif off_k == max_ind:
            seq_start = tl.load(seq_offsets + off_b)
            seq_end = tl.load(seq_offsets + off_b + 1)
            for k in range(seq_start + max_ind, seq_end):
                jagged_ptr = Jagged + k * stride_jn
                jagged_ptrs = jagged_ptr + offs_d
                jg = tl.load(jagged_ptrs, mask=offs_d < D)
                accumulator += jg
                if SCALE_JAGGED:
                    out_jagged_ptr = JaggedOut + k * stride_jon
                    out_jagged_ptrs = out_jagged_ptr + offs_d
                    tl.store(out_jagged_ptrs, jg * scale, mask=offs_d < D)
    out = accumulator
    out_ptrs = DenseOut + off_k * stride_don + offs_d
    tl.store(out_ptrs, out, mask=offs_d < D)
