import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_splitK(Q, K, V, sm_scale, Out_splitK, LSE_splitk,
    block_tables, Seq_len, Seq_starts_k, Seq_starts_q,
    Seq_starts_q_multiplier, additive_bias, K_fp8_scale_shift,
    V_fp8_scale_shift, stride_qz, stride_qm, stride_qg, stride_qh,
    stride_qk, stride_kz, stride_kn, stride_kg, stride_kh, stride_kk,
    stride_vz, stride_vn, stride_vg, stride_vh, stride_vk, stride_osk_z,
    stride_osk_g, stride_osk_h, stride_osk_s, stride_osk_m, stride_osk_k,
    stride_lsek_z, stride_lsek_g, stride_lsek_h, stride_lsek_s,
    stride_lsek_m, stride_blocktablesz, stride_blocktablesl, stride_bias_b,
    stride_bias_g, stride_bias_h, stride_bias_qm, stride_bias_km,
    stride_k_fp8_scale_shift_z: 'tl.constexpr', stride_k_fp8_scale_shift_n:
    'tl.constexpr', stride_k_fp8_scale_shift_g: 'tl.constexpr',
    stride_k_fp8_scale_shift_h: 'tl.constexpr', stride_v_fp8_scale_shift_z:
    'tl.constexpr', stride_v_fp8_scale_shift_n: 'tl.constexpr',
    stride_v_fp8_scale_shift_g: 'tl.constexpr', stride_v_fp8_scale_shift_h:
    'tl.constexpr', kv_cache_blocks_per_row: 'tl.constexpr', Z:
    'tl.constexpr', N_CTX_Q: 'tl.constexpr', N_CTX_K: 'tl.constexpr',
    BLOCK_N_PER_SPLIT: 'tl.constexpr', H: 'tl.constexpr', G: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr', USE_SEQ_LEN: 'tl.constexpr',
    PACKED_PER_VAL: 'tl.constexpr', N_GROUPS: 'tl.constexpr',
    BOUNDS_CHECKS_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N:
    'tl.constexpr', IS_SPLITK: 'tl.constexpr', SPLIT_K_EARLY_EXIT:
    'tl.constexpr', IS_CAUSAL: 'tl.constexpr', NUM_QUERIES_CAUSAL:
    'tl.constexpr', USE_PAGED_ATTENTION: 'tl.constexpr', PAGE_SIZE:
    'tl.constexpr', WRITE_LSE: 'tl.constexpr', HAS_ADDITIVE_BIAS:
    'tl.constexpr'):
    """This kernel can accept non-quantized or int4-quantized keys/values.
    PACKED_PER_VAL determines the quantization type:
        - PACKED_PER_VAL == 1 means no quantization
        - PACKED_PER_VAL == 8 means 4-bit quantization (8 packed quantized values inside one int32)
    For the quantized case K/V should be int32 tensors.
    Quantization can be row-wise (when N_GROUPS = 1) or group-wise with N_GROUPS = 2, 4, or 8.
    Quantization coefficients are stored at the beginning of the row along the last dimension of K/V
    So K[B, H, M, :] has a form
    [   quant_coef0, quant_coef1, ...|
        group0_quant_value0, group0_quant_value1,... |
        group1_quant_value0, group1_quant_value1,...]
    where each quant_coef is an int32 which should be interpreted as 2 packed float16: scale and offset.

    Note: this kernel needs to be processed by xformers.triton.vararg_kernel.unroll_varargs
    before compilation. That will unroll variables marked with "VAR_ARGS_ARRAY" into lists.
    See how FwOp.apply does it below.

    Set IS_SPLITK=False to indicate the MHA result should be written directly.
    No metadata will be written.
    """
    internal_dtype = (tl.float64 if Out_splitK.dtype.element_ty is tl.
        float64 else tl.float32)
    tl.static_assert(PACKED_PER_VAL == 1 and tl.constexpr(K.dtype.
        element_ty != tl.int32) or (PACKED_PER_VAL == 4 or PACKED_PER_VAL ==
        8) and tl.constexpr(K.dtype.element_ty == tl.int32),
        f'Only int4 and fp8 quantization is supported, K/V should have dtype int32 in the quantized case: PACKED_PER_VAL={PACKED_PER_VAL!r} tl.constexpr(K.dtype)={tl.constexpr(K.dtype)!r} tl.constexpr(K.dtype.element_ty)={tl.constexpr(K.dtype.element_ty)!r}'
        )
    tl.static_assert(((N_GROUPS == 1 or N_GROUPS == 2) or N_GROUPS == 4) or
        N_GROUPS == 8,
        'Number of quantization groups can be 1 (row-wise quantization), 2, 4, or 8.'
        )
    tl.static_assert(N_GROUPS == 1 or K_fp8_scale_shift is None,
        f'Only row-wise fp8 quantization is supported, but got N_GROUPS={N_GROUPS!r} > 1.'
        )
    FP8_QUANTIZED: 'tl.constexpr' = K_fp8_scale_shift is not None
    INT4_QUANTIZED: 'tl.constexpr' = PACKED_PER_VAL > 1 and not FP8_QUANTIZED
    PACKED_D_PER_GROUP: 'tl.constexpr' = (BLOCK_DMODEL // PACKED_PER_VAL //
        N_GROUPS)
    D_PER_GROUP: 'tl.constexpr' = BLOCK_DMODEL // N_GROUPS
    start_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_hg = off_zhg % (H * G)
    off_h = off_hg // G
    off_g = off_hg % G
    splitk_idx = tl.program_id(2)
    if USE_SEQ_LEN:
        kv_len = tl.load(Seq_len + off_z)
        if SPLIT_K_EARLY_EXIT and kv_len == 0:
            return
    else:
        kv_len = N_CTX_K
    if Seq_starts_k is None:
        start_kv_idx = 0
    else:
        start_kv_idx = tl.load(Seq_starts_k + off_z)
    if Seq_starts_q is None:
        q_len = N_CTX_Q
        queries_use_batch_dim = 1
        off_m = 0
    else:
        queries_use_batch_dim = 0
        off_m = tl.load(Seq_starts_q + off_z) * Seq_starts_q_multiplier
        q_len = tl.load(Seq_starts_q + off_z + 1
            ) * Seq_starts_q_multiplier - off_m
        if q_len == 0:
            return
    k_base = K + off_h * stride_kh + off_g * stride_kg
    v_base = V + off_h * stride_vh + off_g * stride_vg
    if FP8_QUANTIZED:
        k_fp8_scale_shift_base = (K_fp8_scale_shift + off_h *
            stride_k_fp8_scale_shift_h + off_g * stride_k_fp8_scale_shift_g)
        v_fp8_scale_shift_base = (V_fp8_scale_shift + off_h *
            stride_v_fp8_scale_shift_h + off_g * stride_v_fp8_scale_shift_g)
    else:
        k_fp8_scale_shift_base = None
        v_fp8_scale_shift_base = None
    chunk_hi = (splitk_idx + 1) * BLOCK_N_PER_SPLIT
    chunk_lo = splitk_idx * BLOCK_N_PER_SPLIT
    ignore_in_first_block = 0
    if PAGE_SIZE > 0:
        BLOCKS_IN_PAGE: 'tl.constexpr' = PAGE_SIZE // BLOCK_N
        is_last_chunk = splitk_idx == tl.num_programs(2) - 1
        shift = BLOCK_N - 1 if is_last_chunk else 0
        lo = tl.maximum(chunk_lo, start_kv_idx) // BLOCK_N * BLOCK_N
        ignore_in_first_block = tl.maximum(0, start_kv_idx - lo)
        hi = (chunk_hi + shift) // BLOCK_N * BLOCK_N
        hi = tl.minimum(hi, kv_len + start_kv_idx)
        block_table = block_tables + stride_blocktablesz * off_z
        logical_block_idx = lo // BLOCK_N
    else:
        lo = chunk_lo
        hi = tl.minimum(chunk_hi, kv_len)
        if Seq_starts_k is not None:
            k_base += start_kv_idx * stride_kn
            v_base += start_kv_idx * stride_vn
        else:
            k_base += off_z * stride_kz
            v_base += off_z * stride_vz
        K_block_ptr = tl.make_block_ptr(base=k_base + stride_kk *
            INT4_QUANTIZED * N_GROUPS, shape=(PACKED_D_PER_GROUP, hi),
            strides=(stride_kk, stride_kn), offsets=(0, lo), block_shape=(
            PACKED_D_PER_GROUP, BLOCK_N), order=(0, 1))
        V_block_ptr = tl.make_block_ptr(base=v_base + stride_vk *
            INT4_QUANTIZED * N_GROUPS, shape=(hi, PACKED_D_PER_GROUP),
            strides=(stride_vn, stride_vk), offsets=(lo, 0), block_shape=(
            BLOCK_N, PACKED_D_PER_GROUP), order=(1, 0))
        if INT4_QUANTIZED:
            K_scale_shift_block_ptr = tl.make_block_ptr(base=k_base, shape=
                (1, hi), strides=(stride_kk, stride_kn), offsets=(0, lo),
                block_shape=(1, BLOCK_N), order=(0, 1))
            V_scale_shift_block_ptr = tl.make_block_ptr(base=v_base, shape=
                (hi, 1), strides=(stride_vn, stride_vk), offsets=(lo, 0),
                block_shape=(BLOCK_N, 1), order=(1, 0))
        elif FP8_QUANTIZED:
            if Seq_starts_k is not None:
                k_fp8_scale_shift_base += (start_kv_idx *
                    stride_k_fp8_scale_shift_n)
                v_fp8_scale_shift_base += (start_kv_idx *
                    stride_v_fp8_scale_shift_n)
            else:
                k_fp8_scale_shift_base += off_z * stride_k_fp8_scale_shift_z
                v_fp8_scale_shift_base += off_z * stride_v_fp8_scale_shift_z
            K_scale_shift_block_ptr = tl.make_block_ptr(base=
                k_fp8_scale_shift_base, shape=(1, hi), strides=(1,
                stride_k_fp8_scale_shift_n), offsets=(0, lo), block_shape=(
                1, BLOCK_N), order=(0, 1))
            V_scale_shift_block_ptr = tl.make_block_ptr(base=
                v_fp8_scale_shift_base, shape=(hi, 1), strides=(
                stride_v_fp8_scale_shift_n, 1), offsets=(lo, 0),
                block_shape=(BLOCK_N, 1), order=(1, 0))
        else:
            K_scale_shift_block_ptr = None
            V_scale_shift_block_ptr = None
        if HAS_ADDITIVE_BIAS:
            additive_bias_block_ptr = tl.make_block_ptr(base=additive_bias +
                off_z * stride_bias_b + off_g * stride_bias_g + off_h *
                stride_bias_h, shape=(N_CTX_Q, hi), strides=(stride_bias_qm,
                stride_bias_km), offsets=(start_m * BLOCK_M, lo),
                block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))
    if SPLIT_K_EARLY_EXIT and lo >= hi:
        return
    Q_block_ptr = tl.make_block_ptr(base=Q + off_m * stride_qm + off_h *
        stride_qh + off_z * stride_qz * queries_use_batch_dim + off_g *
        stride_qg, shape=(q_len, BLOCK_DMODEL), strides=(stride_qm,
        stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M,
        D_PER_GROUP), order=(1, 0))
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc: "'VAR_ARGS_ARRAY'"
    for i in range(len(acc)):
        acc[i] = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=internal_dtype)
    qk_scale = sm_scale * 1.44269504
    q: "'VAR_ARGS_ARRAY'"
    for i in range(len(acc)):
        q[i] = tl.load(tl.advance(Q_block_ptr, (0, i * D_PER_GROUP)),
            boundary_check=(0,))
    if IS_CAUSAL:
        q_offset = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        diag_idx = q_offset[:, None] % NUM_QUERIES_CAUSAL - tl.arange(0,
            BLOCK_N)[None, :]
        diag_idx_shifted = tl.constexpr(diag_idx - NUM_QUERIES_CAUSAL + kv_len)
    for start_n in range(lo, hi, BLOCK_N):
        if PAGE_SIZE > 0:
            block_offset_in_page = logical_block_idx % BLOCKS_IN_PAGE
            logical_page_idx = logical_block_idx // BLOCKS_IN_PAGE
            physical_page_idx = tl.load(block_table + stride_blocktablesl *
                logical_page_idx)
            offset = (physical_page_idx * PAGE_SIZE + block_offset_in_page *
                BLOCK_N)
            current_block_size = min(hi - start_n, BLOCK_N)
            K_block_ptr = tl.make_block_ptr(base=k_base + stride_kk *
                INT4_QUANTIZED * N_GROUPS, shape=(PACKED_D_PER_GROUP, 
                offset + current_block_size), strides=(stride_kk, stride_kn
                ), offsets=(0, offset), block_shape=(PACKED_D_PER_GROUP,
                BLOCK_N), order=(0, 1))
            V_block_ptr = tl.make_block_ptr(base=v_base + stride_vk *
                INT4_QUANTIZED * N_GROUPS, shape=(offset +
                current_block_size, PACKED_D_PER_GROUP), strides=(stride_vn,
                stride_vk), offsets=(offset, 0), block_shape=(BLOCK_N,
                PACKED_D_PER_GROUP), order=(1, 0))
            if INT4_QUANTIZED:
                K_scale_shift_block_ptr = tl.make_block_ptr(base=k_base,
                    shape=(1, offset + current_block_size), strides=(
                    stride_kk, stride_kn), offsets=(0, offset), block_shape
                    =(1, BLOCK_N), order=(0, 1))
                V_scale_shift_block_ptr = tl.make_block_ptr(base=v_base,
                    shape=(offset + current_block_size, 1), strides=(
                    stride_vn, stride_vk), offsets=(offset, 0), block_shape
                    =(BLOCK_N, 1), order=(1, 0))
            elif FP8_QUANTIZED:
                K_scale_shift_block_ptr = tl.make_block_ptr(base=
                    k_fp8_scale_shift_base, shape=(1, offset +
                    current_block_size), strides=(1,
                    stride_k_fp8_scale_shift_n), offsets=(0, offset),
                    block_shape=(1, BLOCK_N), order=(0, 1))
                V_scale_shift_block_ptr = tl.make_block_ptr(base=
                    v_fp8_scale_shift_base, shape=(offset +
                    current_block_size, 1), strides=(
                    stride_v_fp8_scale_shift_n, 1), offsets=(offset, 0),
                    block_shape=(BLOCK_N, 1), order=(1, 0))
            else:
                K_scale_shift_block_ptr = None
                V_scale_shift_block_ptr = None
            logical_block_idx += 1
        k: "'VAR_ARGS_ARRAY'"
        v: "'VAR_ARGS_ARRAY'"
        for i in range(len(acc)):
            k[i], v[i] = load_dequantize_k_v_group(K_block_ptr, V_block_ptr,
                K_scale_shift_block_ptr, V_scale_shift_block_ptr,
                BOUNDS_CHECKS_N, PACKED_PER_VAL, PACKED_D_PER_GROUP,
                FP8_QUANTIZED, Q.dtype.element_ty, i)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for i in range(len(acc)):
            qk += tl.dot(q[i], k[i])
        qk *= qk_scale
        if start_n == lo and ignore_in_first_block > 0:
            qk = tl.where(tl.arange(0, BLOCK_N) < ignore_in_first_block,
                float('-inf'), qk)
        if HAS_ADDITIVE_BIAS:
            loaded_bias = tl.load(additive_bias_block_ptr, boundary_check=(
                0, 1) if BOUNDS_CHECKS_N else (0,))
            qk += loaded_bias * 1.44269504
            additive_bias_block_ptr = tl.advance(additive_bias_block_ptr, (
                0, BLOCK_N))
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float(
                '-inf'))
        if IS_CAUSAL:
            qk = tl.where(diag_idx_shifted >= start_n, qk, float('-inf'))
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        if HAS_ADDITIVE_BIAS or IS_CAUSAL:
            alpha = tl.where(m_i_new == float('-inf'), 0, alpha)
            p = tl.where(m_i_new[:, None] == float('-inf'), 0, p)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p
        for i in range(len(acc)):
            acc[i] *= alpha[:, None]
            acc[i] += tl.dot(p, v[i])
        if not PAGE_SIZE:
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            if PACKED_PER_VAL > 1:
                K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr,
                    (0, BLOCK_N))
                V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr,
                    (BLOCK_N, 0))
    O_block_ptr = tl.make_block_ptr(base=Out_splitK + off_z * stride_osk_z *
        queries_use_batch_dim + off_m * stride_osk_m + off_g * stride_osk_g +
        off_h * stride_osk_h + splitk_idx * stride_osk_s, shape=(q_len,
        D_PER_GROUP), strides=(stride_osk_m, 1), offsets=(start_m * BLOCK_M,
        0), block_shape=(BLOCK_M, D_PER_GROUP), order=(1, 0))
    for i in range(len(acc)):
        attn_out = tl.where(l_i[:, None] == 0, 0.0, acc[i] / l_i[:, None])
        tl.store(tl.advance(O_block_ptr, (0, i * D_PER_GROUP)), attn_out,
            boundary_check=(0,))
    if WRITE_LSE:
        LSE_splitk_ptr = (LSE_splitk + off_z * stride_lsek_z *
            queries_use_batch_dim + off_m * stride_lsek_m + off_g *
            stride_lsek_g + off_h * stride_lsek_h + splitk_idx *
            stride_lsek_s + (start_m * BLOCK_M + tl.arange(0, BLOCK_M)) *
            stride_lsek_m)
        mask = start_m * BLOCK_M + tl.arange(0, BLOCK_M) < q_len
        lse_dtype = LSE_splitk.dtype.element_ty
        tl.store(LSE_splitk_ptr, (tl.math.log2(l_i) + m_i) / 1.44269504,
            mask=mask)
