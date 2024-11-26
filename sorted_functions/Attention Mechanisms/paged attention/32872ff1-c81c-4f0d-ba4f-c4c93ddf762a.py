import triton
import triton.language as tl
import torch

@triton.jit
def _inner_paged_attn_unroll_4_kernel(q, k_cache, v_cache, stride_km,
    block_base_ptrs, base_offs_kv, alibi_slope, block_offs, seq_len, qkv,
    qk_max, exp_sum, BLOCK_SIZE: 'tl.constexpr', LO: 'tl.constexpr', HI:
    'tl.constexpr'):
    for block_idx in range(LO, HI, 4):
        offs_kv_0 = tl.load(block_base_ptrs + block_idx + 0
            ) * stride_km + base_offs_kv
        offs_kv_1 = tl.load(block_base_ptrs + block_idx + 1
            ) * stride_km + base_offs_kv
        offs_kv_2 = tl.load(block_base_ptrs + block_idx + 2
            ) * stride_km + base_offs_kv
        offs_kv_3 = tl.load(block_base_ptrs + block_idx + 3
            ) * stride_km + base_offs_kv
        k_0 = tl.load(k_cache + offs_kv_0)
        k_1 = tl.load(k_cache + offs_kv_1)
        k_2 = tl.load(k_cache + offs_kv_2)
        k_3 = tl.load(k_cache + offs_kv_3)
        v_0 = tl.load(v_cache + offs_kv_0)
        v_1 = tl.load(v_cache + offs_kv_1)
        v_2 = tl.load(v_cache + offs_kv_2)
        v_3 = tl.load(v_cache + offs_kv_3)
        _qk_0 = tl.sum(q[None, :] * k_0, axis=1)
        _qk_1 = tl.sum(q[None, :] * k_1, axis=1)
        _qk_2 = tl.sum(q[None, :] * k_2, axis=1)
        _qk_3 = tl.sum(q[None, :] * k_3, axis=1)
        if alibi_slope is not None:
            _qk_0 += alibi_slope * ((block_idx + 0) * BLOCK_SIZE +
                block_offs - seq_len + 1)
            _qk_1 += alibi_slope * ((block_idx + 1) * BLOCK_SIZE +
                block_offs - seq_len + 1)
            _qk_2 += alibi_slope * ((block_idx + 2) * BLOCK_SIZE +
                block_offs - seq_len + 1)
            _qk_3 += alibi_slope * ((block_idx + 3) * BLOCK_SIZE +
                block_offs - seq_len + 1)
        _qk_max = tl.maximum(tl.max(_qk_0, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_1, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_2, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_3, axis=0), _qk_max)
        exp_tmp = tl.exp(_qk_0 - _qk_max) + tl.exp(_qk_1 - _qk_max) + tl.exp(
            _qk_2 - _qk_max) + tl.exp(_qk_3 - _qk_max)
        _exp_sum = exp_sum * tl.exp(qk_max - _qk_max) + tl.sum(exp_tmp, axis=0)
        qkv_sum_tmp = tl.exp(_qk_0[:, None] - _qk_max) * v_0 + tl.exp(_qk_1
            [:, None] - _qk_max) * v_1 + tl.exp(_qk_2[:, None] - _qk_max
            ) * v_2 + tl.exp(_qk_3[:, None] - _qk_max) * v_3
        qkv = (qkv * (exp_sum * tl.exp(qk_max - _qk_max)) + qkv_sum_tmp
            ) / _exp_sum
        qk_max = _qk_max
        exp_sum = _exp_sum
    return qkv, qk_max, exp_sum
