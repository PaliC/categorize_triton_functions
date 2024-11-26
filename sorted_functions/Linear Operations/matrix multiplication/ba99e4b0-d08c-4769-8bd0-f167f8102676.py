import triton
import triton.language as tl
import torch

@triton.heuristics({'EVEN_M': lambda args: args['seqlen'] % args['BLOCK_M'] ==
    0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM']})
@triton.jit
def _angular_lsh_kernel(in_mat, proj_dir, perm, enc_vec, buckets,
    stride_in_matb, stride_in_math, stride_in_matm, stride_proj_dirb,
    stride_proj_dirh, stride_proj_dird, stride_bucketsb, stride_bucketsh,
    nheads, seqlen, seqlen_rounded, headdim, NUM_PROJ_ROUNDED:
    'tl.constexpr', num_projs: 'tl.constexpr', BLOCK_HEADDIM:
    'tl.constexpr', EVEN_M: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr',
    BLOCK_M: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, NUM_PROJ_ROUNDED)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    in_mat_ptrs = in_mat + off_b * stride_in_matb + off_h * stride_in_math + (
        offs_m[:, None] * stride_in_matm + offs_d[None, :])
    proj_dir_ptrs = (proj_dir + off_b * stride_proj_dirb + off_h *
        stride_proj_dirh + (offs_d[:, None] * stride_proj_dird + offs_n[
        None, :]))
    if EVEN_M:
        if EVEN_HEADDIM:
            mat = tl.load(in_mat_ptrs)
        else:
            mat = tl.load(in_mat_ptrs, mask=offs_d[None, :] < headdim,
                other=0.0)
    elif EVEN_HEADDIM:
        mat = tl.load(in_mat_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    else:
        mat = tl.load(in_mat_ptrs, mask=(offs_m[:, None] < seqlen) & (
            offs_d[None, :] < headdim), other=0.0)
    if EVEN_HEADDIM:
        proj_dir_block = tl.load(proj_dir_ptrs, mask=offs_n[None, :] <
            num_projs, other=0.0)
    else:
        proj_dir_block = tl.load(proj_dir_ptrs, mask=(offs_n[None, :] <
            num_projs) & (offs_d[:, None] * stride_proj_dird < headdim),
            other=0.0)
    mask = tl.dot(mat, proj_dir_block)
    mask = tl.where(mask > 0.0, 1.0, 0.0)
    encoding_vectors = tl.load(enc_vec + offs_n, mask=offs_n < num_projs,
        other=0.0)
    bin_ids = tl.sum(mask * encoding_vectors[None, :], 1)
    hash_buckets = tl.load(perm + bin_ids)
    buckets_ptrs = (buckets + off_b * stride_bucketsb + off_h *
        stride_bucketsh + offs_m)
    if EVEN_M:
        tl.store(buckets_ptrs, hash_buckets)
    else:
        tl.store(buckets_ptrs, hash_buckets, mask=offs_m < seqlen)
