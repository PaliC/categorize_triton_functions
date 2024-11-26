import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, bias_ptr, mask_ptr, k_start_ptr,
    k_end_ptr, q_offset, k_offset, v_offset, q_stride_b, q_stride_s,
    q_stride_h, q_stride_d, k_stride_b, k_stride_s, k_stride_h, k_stride_d,
    v_stride_b, v_stride_s, v_stride_h, v_stride_d, bias_stride_b,
    bias_stride_h, bias_stride_sq, bias_stride_sk, mask_stride_b,
    mask_stride_h, mask_stride_sq, mask_stride_sk, k_start_stride_b,
    k_start_stride_h, k_start_stride_sq, k_end_stride_b, k_end_stride_h,
    k_end_stride_sq, o_stride_b, o_stride_s, o_stride_h, o_stride_d,
    num_heads_q, num_heads_k, seq_len_q, seq_len_k, o_ptr, is_causal:
    'tl.constexpr', use_attention_mask: 'tl.constexpr', use_k_start:
    'tl.constexpr', use_k_end: 'tl.constexpr', use_bias: 'tl.constexpr',
    sm_scale: 'tl.constexpr', block_q: 'tl.constexpr', block_k:
    'tl.constexpr', head_dim: 'tl.constexpr', use_mask_q: 'tl.constexpr',
    use_mask_k: 'tl.constexpr', bias_bcast_sq: 'tl.constexpr',
    mask_bcast_sq: 'tl.constexpr', dot_fn_qk: 'tl.constexpr', dot_fn_kv:
    'tl.constexpr'):
    """Triton MHA forward kernel."""
    block_d: 'tl.constexpr' = jt.utils.next_power_of_2(head_dim.value)
    start_q = tl.program_id(1) * block_q
    off_h = tl.program_id(0)
    off_b = tl.program_id(2)
    off_h_k = off_h // (num_heads_q // num_heads_k)
    q_ptr += off_h * q_stride_h + off_b * q_stride_b + q_offset
    k_ptr += off_h_k * k_stride_h + off_b * k_stride_b + k_offset
    v_ptr += off_h_k * v_stride_h + off_b * v_stride_b + v_offset
    o_ptr += off_h * o_stride_h + off_b * o_stride_b
    if use_bias:
        bias_ptr += off_b * bias_stride_b + off_h * bias_stride_h
    if use_attention_mask:
        mask_ptr += off_b * mask_stride_b + off_h * mask_stride_h
    if use_k_start:
        k_start_ptr += off_b * k_start_stride_b + off_h * k_start_stride_h
    if use_k_end:
        k_end_ptr += off_b * k_end_stride_b + off_h * k_end_stride_h
    q_block_ptr = tl.make_block_ptr(q_ptr, shape=(seq_len_q, head_dim),
        strides=(q_stride_s, q_stride_d), offsets=(start_q, 0), block_shape
        =(block_q, block_d), order=(1, 0))
    k_block_ptr = tl.make_block_ptr(k_ptr, shape=(head_dim, seq_len_k),
        strides=(k_stride_d, k_stride_s), offsets=(0, 0), block_shape=(
        block_d, block_k), order=(0, 1))
    v_block_ptr = tl.make_block_ptr(v_ptr, shape=(seq_len_k, head_dim),
        strides=(v_stride_s, v_stride_d), offsets=(0, 0), block_shape=(
        block_k, block_d), order=(1, 0))
    q_boundary_check0: 'tl.constexpr' = (0,) if use_mask_q else ()
    q_boundary_check1: 'tl.constexpr' = (1,) if head_dim != block_d else ()
    q_boundary_check: 'tl.constexpr' = q_boundary_check0 + q_boundary_check1
    q_padding_option: 'tl.constexpr' = 'zero' if len(q_boundary_check.value
        ) else ''
    k_boundary_check: 'tl.constexpr' = (0,) if head_dim != block_d else ()
    v_boundary_check: 'tl.constexpr' = (0,) if use_mask_k else ()
    bias_start_dim: 'tl.constexpr' = 1 if bias_bcast_sq else 0
    bias_block_ptr = tl.make_block_ptr(bias_ptr, shape=(seq_len_q,
        seq_len_k)[bias_start_dim:], strides=(bias_stride_sq,
        bias_stride_sk)[bias_start_dim:], offsets=(start_q, 0)[
        bias_start_dim:], block_shape=(block_q, block_k)[bias_start_dim:],
        order=(1, 0)[bias_start_dim:])
    bias_advance: 'tl.constexpr' = (0, block_k)[bias_start_dim:]
    mask_start_dim: 'tl.constexpr' = 1 if mask_bcast_sq else 0
    mask_block_ptr = tl.make_block_ptr(mask_ptr, shape=(seq_len_q,
        seq_len_k)[mask_start_dim:], strides=(mask_stride_sq,
        mask_stride_sk)[mask_start_dim:], offsets=(start_q, 0)[
        mask_start_dim:], block_shape=(block_q, block_k)[mask_start_dim:],
        order=(1, 0)[mask_start_dim:])
    mask_advance: 'tl.constexpr' = (0, block_k)[mask_start_dim:]
    k_start_block_ptr = tl.make_block_ptr(k_start_ptr, shape=(seq_len_q,),
        strides=(k_start_stride_sq,), offsets=(start_q,), block_shape=(
        block_q,), order=(0,))
    k_end_block_ptr = tl.make_block_ptr(k_end_ptr, shape=(seq_len_q,),
        strides=(k_end_stride_sq,), offsets=(start_q,), block_shape=(
        block_q,), order=(0,))
    span_q = start_q + tl.arange(0, block_q)
    m_i = tl.full([block_q], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([block_q], dtype=tl.float32)
    acc = tl.zeros([block_q, block_d], dtype=tl.float32)
    q = tl.load(q_block_ptr, boundary_check=q_boundary_check,
        padding_option=q_padding_option)
    q *= sm_scale
    k_start = None
    if use_k_start:
        k_start = tl.load(k_start_block_ptr)
        start_loop = tl.maximum(tl.min(k_start), 0)
        blocks_to_skip = start_loop // block_k
        start_loop = block_k * blocks_to_skip
        for _ in range(blocks_to_skip):
            k_block_ptr = tl.advance(k_block_ptr, (0, block_k))
            v_block_ptr = tl.advance(v_block_ptr, (block_k, 0))
            bias_block_ptr = tl.advance(bias_block_ptr, bias_advance.value)
            mask_block_ptr = tl.advance(mask_block_ptr, mask_advance.value)
    else:
        start_loop = 0
    k_end = None
    if is_causal:
        end_loop = tl.minimum(start_q // block_k * block_k, seq_len_k)
    elif use_k_end:
        k_end = tl.load(k_end_block_ptr)
        end_loop = tl.minimum(tl.max(k_end), seq_len_k)
    else:
        end_loop = seq_len_k
    (k_block_ptr, v_block_ptr, bias_block_ptr, mask_block_ptr, acc, m_i, l_i
        ) = (_fwd_kernel_inner(start_loop, end_loop, q, span_q, k_block_ptr,
        v_block_ptr, bias_block_ptr, mask_block_ptr, k_start, k_end,
        seq_len_k, acc, m_i, l_i, bias_advance, mask_advance, False,
        use_attention_mask, use_k_start, use_k_end, use_bias, block_k,
        use_mask_k, k_boundary_check, v_boundary_check, dot_fn_qk, dot_fn_kv))
    if is_causal:
        tl.debug_barrier()
        start_loop, end_loop = end_loop, tl.minimum(end_loop + block_k,
            seq_len_k)
        _, _, _, _, acc, _, l_i = _fwd_kernel_inner(start_loop, end_loop, q,
            span_q, k_block_ptr, v_block_ptr, bias_block_ptr,
            mask_block_ptr, k_start, k_end, seq_len_k, acc, m_i, l_i,
            bias_advance, mask_advance, True, use_attention_mask,
            use_k_start, use_k_end, use_bias, block_k, use_mask_k,
            k_boundary_check, v_boundary_check, dot_fn_qk, dot_fn_kv)
    l_i += float(jnp.finfo(jnp.float32).tiny)
    acc /= l_i[:, None]
    o_block_ptr = tl.make_block_ptr(o_ptr, shape=(seq_len_q, head_dim),
        strides=(o_stride_s, o_stride_d), offsets=(start_q, 0), block_shape
        =(block_q, block_d), order=(1, 0))
    acc = acc
    tl.store(o_block_ptr, acc, boundary_check=q_boundary_check)
