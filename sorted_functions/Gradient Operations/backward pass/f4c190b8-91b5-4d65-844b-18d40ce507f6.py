import triton
import triton.language as tl
import torch

@triton.jit
def bwd_kernel_dk_dv(Q, K, V, B, sm_scale, Out, DO, DK, DV, L, D, stride_qz,
    stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn,
    stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_bz,
    stride_bh, stride_bm, stride_bn, stride_oz, stride_oh, stride_om,
    stride_ok, stride_dkz, stride_dkh, stride_dkn, stride_dkk, stride_dvz,
    stride_dvh, stride_dvk, stride_dvn, num_head_q: "'i32'", num_head_k:
    "'i32'", cu_seqlens_q, cu_seqlens_k, num_seqlens: "'i32'", max_seqlen_q:
    "'i32'", max_seqlen_k: "'i32'", head_dim: "'i32'", dropout_p,
    philox_seed_ptr, philox_offset1: "'*u32'", philox_offset2: "'u32'",
    BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', CAUSAL: 'tl.constexpr', ENABLE_DROPOUT: 'tl.constexpr',
    PADDED_HEAD: 'tl.constexpr', BIAS_TYPE: 'tl.constexpr'):
    philox_seed = 0
    philox_offset_base = philox_offset2
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
    start_k = tl.program_id(0) * BLOCK_N
    off_h_k = tl.program_id(1)
    off_z = tl.program_id(2)
    num_z = tl.num_programs(2)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_k + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    ld_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
    cu_seqlens_q_start = 0
    cu_seqlens_k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    batch_index = off_z
    if num_seqlens > 0:
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        if start_k >= seqlen_k:
            return
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        batch_index = 0
    if num_seqlens < 0:
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        if start_k >= seqlen_k:
            return
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        batch_index = off_z
    k_offset = (off_h_k * stride_kh + batch_index * stride_kz + 
        cu_seqlens_k_start * stride_kn)
    K += k_offset
    kt_ptrs = K + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    if start_k + BLOCK_N <= seqlen_k:
        kt = load_fn(kt_ptrs, ld_offs_d, None, head_dim, seqlen_k)
    else:
        kt = load_fn(kt_ptrs, ld_offs_d, offs_n, head_dim, seqlen_k)
    v_offset = (off_h_k * stride_vh + batch_index * stride_vz + 
        cu_seqlens_k_start * stride_vk)
    V += v_offset
    vt_ptrs = V + offs_d[:, None] * stride_vn + offs_n[None, :] * stride_vk
    if start_k + BLOCK_N <= seqlen_k:
        vt = load_fn(vt_ptrs, ld_offs_d, None, head_dim, seqlen_k)
    else:
        vt = load_fn(vt_ptrs, ld_offs_d, offs_n, head_dim, seqlen_k)
    if BIAS_TYPE == 0:
        B_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(base=B + off_h_k * stride_bh + 
            batch_index * stride_bz, shape=(seqlen_q, seqlen_k), strides=(
            stride_bm, stride_bn), offsets=(0, start_k), block_shape=(
            BLOCK_M, BLOCK_N), order=(1, 0))
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
    dk_offset = (off_h_k * stride_dkh + batch_index * stride_dkz + 
        cu_seqlens_k_start * stride_dkn)
    DK += dk_offset
    dv_offset = (off_h_k * stride_dvh + batch_index * stride_dvz + 
        cu_seqlens_k_start * stride_dvk)
    DV += dv_offset
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504089
    bias_scale = 1.0 / sm_scale
    group_size = num_head_q // num_head_k
    q_lo = start_k if CAUSAL else 0
    q_hi = seqlen_q
    real_seqlen_q = q_hi - q_lo
    n_blocks = tl.cdiv(q_hi - q_lo, BLOCK_M)
    n_extra_tokens = 0
    if real_seqlen_q < BLOCK_M:
        n_extra_tokens = BLOCK_M - real_seqlen_q
    elif real_seqlen_q % BLOCK_M:
        n_extra_tokens = real_seqlen_q % BLOCK_M
    is_irregular_q = n_extra_tokens != 0
    leading_masked_blocks = 0
    trailing_masked_blocks = 0
    if CAUSAL:
        leading_masked_blocks = tl.cdiv(BLOCK_N, BLOCK_M)
        trailing_masked_blocks = 1 if is_irregular_q else 0
    else:
        leading_masked_blocks = 0
        trailing_masked_blocks = 1 if is_irregular_q else 0
    n_full_blocks = n_blocks - leading_masked_blocks - trailing_masked_blocks
    for off_h_q in range(off_h_k * group_size, off_h_k * group_size +
        group_size):
        off_zh = off_z * num_head_q + off_h_q * 1
        if ENABLE_DROPOUT:
            batch_philox_offset = (philox_offset_base + off_zh *
                max_seqlen_q * max_seqlen_k)
        else:
            batch_philox_offset = 0
        D_ptrs = D + off_zh * max_seqlen_q
        l_ptrs = L + off_zh * max_seqlen_q
        q_offset = (off_h_q * stride_qh + batch_index * stride_qz + 
            cu_seqlens_q_start * stride_qm)
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :
            ] * stride_qk
        do_offset = (off_h_q * stride_oh + batch_index * stride_oz + 
            cu_seqlens_q_start * stride_om)
        do_ptrs = DO + do_offset + offs_m[:, None] * stride_om + offs_d[None, :
            ] * stride_ok
        lo = 0
        hi = 0
        if leading_masked_blocks > 0:
            lo = q_lo
            hi = lo + leading_masked_blocks * BLOCK_M
            overflow_size = 0 if hi < q_hi else hi - q_hi
            dk, dv = bwd_inner_dk_dv(dk, dv, qk_scale, bias_scale, q_ptrs,
                stride_qm, kt, vt, B_block_ptr, do_ptrs, stride_om, l_ptrs,
                D_ptrs, seqlen_q, seqlen_k, head_dim, start_k, lo, hi,
                overflow_size, dropout_p, philox_seed, batch_philox_offset,
                max_seqlen_k, BLOCK_M, BLOCK_DMODEL, BLOCK_N, False, CAUSAL,
                ENABLE_DROPOUT, PADDED_HEAD, BIAS_TYPE)
            tl.debug_barrier()
        if n_full_blocks > 0:
            lo = q_lo + leading_masked_blocks * BLOCK_M
            hi = lo + n_full_blocks * BLOCK_M
            dk, dv = bwd_inner_dk_dv(dk, dv, qk_scale, bias_scale, q_ptrs,
                stride_qm, kt, vt, B_block_ptr, do_ptrs, stride_om, l_ptrs,
                D_ptrs, seqlen_q, seqlen_k, head_dim, start_k, lo, hi, 0,
                dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
                BLOCK_M, BLOCK_DMODEL, BLOCK_N, True, False, ENABLE_DROPOUT,
                PADDED_HEAD, BIAS_TYPE)
        if n_full_blocks >= 0 and trailing_masked_blocks > 0:
            tl.debug_barrier()
            lo = (q_lo + leading_masked_blocks * BLOCK_M + n_full_blocks *
                BLOCK_M)
            hi = q_hi
            overflow_size = lo + trailing_masked_blocks * BLOCK_M - q_hi
            dk, dv = bwd_inner_dk_dv(dk, dv, qk_scale, bias_scale, q_ptrs,
                stride_qm, kt, vt, B_block_ptr, do_ptrs, stride_om, l_ptrs,
                D_ptrs, seqlen_q, seqlen_k, head_dim, start_k, lo, hi,
                overflow_size, dropout_p, philox_seed, batch_philox_offset,
                max_seqlen_k, BLOCK_M, BLOCK_DMODEL, BLOCK_N, False, CAUSAL,
                ENABLE_DROPOUT, PADDED_HEAD, BIAS_TYPE)
    dk = dk * sm_scale
    dv = dv
    mstore2d(dk, BLOCK_N, BLOCK_DMODEL, o_base=DK, o_start_row=start_k,
        o_start_col=0, o_rows=seqlen_k, o_cols=head_dim, stride_row=
        stride_dkn, stride_col=stride_dkk)
    mstore2d(dv, BLOCK_N, BLOCK_DMODEL, o_base=DV, o_start_row=start_k,
        o_start_col=0, o_rows=seqlen_k, o_cols=head_dim, stride_row=
        stride_dvk, stride_col=stride_dvn)
