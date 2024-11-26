import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_inner(start_loop, end_loop, q, span_q, k_block_ptr,
    v_block_ptr, bias_block_ptr, mask_block_ptr, k_start, k_end, seq_len_k,
    acc, m_i, l_i, bias_advance: 'tl.constexpr', mask_advance:
    'tl.constexpr', is_causal: 'tl.constexpr', use_attention_mask:
    'tl.constexpr', use_k_start: 'tl.constexpr', use_k_end: 'tl.constexpr',
    use_bias: 'tl.constexpr', block_k: 'tl.constexpr', use_mask_k:
    'tl.constexpr', k_boundary_check: 'tl.constexpr', v_boundary_check:
    'tl.constexpr', dot_fn_qk: 'tl.constexpr', dot_fn_kv: 'tl.constexpr'):
    """Triton MHA forward kernel's inner loop."""
    for start_k in range(start_loop, end_loop, block_k):
        start_k = tl.multiple_of(start_k, block_k)
        span_k = start_k + tl.arange(0, block_k)
        k = tl.load(k_block_ptr, boundary_check=k_boundary_check,
            padding_option='zero' if len(k_boundary_check.value) else '')
        v = tl.load(v_block_ptr, boundary_check=v_boundary_check,
            padding_option='zero' if len(v_boundary_check.value) else '')
        if use_bias:
            bias = tl.load(bias_block_ptr)
        qk = dot_fn_qk(q, k)
        if use_bias:
            qk = qk & 4294967295
            qk = qk
            qk += bias
        if use_attention_mask | use_k_start | use_k_end:
            mask_value = float(jnp.finfo(jnp.float32).min)
        if use_attention_mask:
            mask = tl.load(mask_block_ptr)
            qk = tl.where(mask, qk, mask_value)
        if use_k_start:
            if tl.sum(k_start) != 0:
                qk = tl.where(k_start[:, None] <= span_k[None, :], qk,
                    mask_value)
        if is_causal:
            qk = tl.where(span_q[:, None] >= span_k[None, :], qk, float('-inf')
                )
        elif use_k_end:
            qk = tl.where(k_end[:, None] > span_k[None, :], qk, mask_value)
        if use_mask_k:
            qk = tl.where((span_k < seq_len_k)[None, :], qk, float('-inf'))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        m_i = m_ij
        acc *= alpha[:, None]
        l_i *= alpha
        l_i += tl.sum(p, axis=1)
        acc += dot_fn_kv(p, v)
        k_block_ptr = tl.advance(k_block_ptr, (0, block_k))
        v_block_ptr = tl.advance(v_block_ptr, (block_k, 0))
        bias_block_ptr = tl.advance(bias_block_ptr, bias_advance.value)
        mask_block_ptr = tl.advance(mask_block_ptr, mask_advance.value)
    return (k_block_ptr, v_block_ptr, bias_block_ptr, mask_block_ptr, acc,
        m_i, l_i)
