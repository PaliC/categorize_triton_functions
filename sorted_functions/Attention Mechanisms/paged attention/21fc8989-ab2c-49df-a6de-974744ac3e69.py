import triton
import triton.language as tl
import torch

@triton.jit
def _inner_paged_attn_unroll_8_kernel(q, k_cache, v_cache, stride_km,
    block_base_ptrs, base_offs_kv, alibi_slope, block_offs, seq_len, qkv,
    qk_max, exp_sum, BLOCK_SIZE: 'tl.constexpr', LO: 'tl.constexpr', HI:
    'tl.constexpr'):
    for block_idx in range(LO, HI, 8):
        offs_kv_0 = tl.load(block_base_ptrs + block_idx + 0
            ) * stride_km + base_offs_kv
        offs_kv_1 = tl.load(block_base_ptrs + block_idx + 1
            ) * stride_km + base_offs_kv
        offs_kv_2 = tl.load(block_base_ptrs + block_idx + 2
            ) * stride_km + base_offs_kv
        offs_kv_3 = tl.load(block_base_ptrs + block_idx + 3
            ) * stride_km + base_offs_kv
        offs_kv_4 = tl.load(block_base_ptrs + block_idx + 4
            ) * stride_km + base_offs_kv
        offs_kv_5 = tl.load(block_base_ptrs + block_idx + 5
            ) * stride_km + base_offs_kv
        offs_kv_6 = tl.load(block_base_ptrs + block_idx + 6
            ) * stride_km + base_offs_kv
        offs_kv_7 = tl.load(block_base_ptrs + block_idx + 7
            ) * stride_km + base_offs_kv
        k_0 = tl.load(k_cache + offs_kv_0)
        k_1 = tl.load(k_cache + offs_kv_1)
        k_2 = tl.load(k_cache + offs_kv_2)
        k_3 = tl.load(k_cache + offs_kv_3)
        k_4 = tl.load(k_cache + offs_kv_4)
        k_5 = tl.load(k_cache + offs_kv_5)
        k_6 = tl.load(k_cache + offs_kv_6)
        k_7 = tl.load(k_cache + offs_kv_7)
        v_0 = tl.load(v_cache + offs_kv_0)
        v_1 = tl.load(v_cache + offs_kv_1)
        v_2 = tl.load(v_cache + offs_kv_2)
        v_3 = tl.load(v_cache + offs_kv_3)
        v_4 = tl.load(v_cache + offs_kv_4)
        v_5 = tl.load(v_cache + offs_kv_5)
        v_6 = tl.load(v_cache + offs_kv_6)
        v_7 = tl.load(v_cache + offs_kv_7)
        _qk_0 = tl.sum(q[None, :] * k_0, axis=1)
        _qk_1 = tl.sum(q[None, :] * k_1, axis=1)
        _qk_2 = tl.sum(q[None, :] * k_2, axis=1)
        _qk_3 = tl.sum(q[None, :] * k_3, axis=1)
        _qk_4 = tl.sum(q[None, :] * k_4, axis=1)
        _qk_5 = tl.sum(q[None, :] * k_5, axis=1)
        _qk_6 = tl.sum(q[None, :] * k_6, axis=1)
        _qk_7 = tl.sum(q[None, :] * k_7, axis=1)
        if alibi_slope is not None:
            _qk_0 += alibi_slope * ((block_idx + 0) * BLOCK_SIZE +
                block_offs - seq_len + 1)
            _qk_1 += alibi_slope * ((block_idx + 1) * BLOCK_SIZE +
                block_offs - seq_len + 1)
            _qk_2 += alibi_slope * ((block_idx + 2) * BLOCK_SIZE +
                block_offs - seq_len + 1)
            _qk_3 += alibi_slope * ((block_idx + 3) * BLOCK_SIZE +
                block_offs - seq_len + 1)
            _qk_4 += alibi_slope * ((block_idx + 4) * BLOCK_SIZE +
                block_offs - seq_len + 1)
            _qk_5 += alibi_slope * ((block_idx + 5) * BLOCK_SIZE +
                block_offs - seq_len + 1)
            _qk_6 += alibi_slope * ((block_idx + 6) * BLOCK_SIZE +
                block_offs - seq_len + 1)
            _qk_7 += alibi_slope * ((block_idx + 7) * BLOCK_SIZE +
                block_offs - seq_len + 1)
        _qk_max = tl.maximum(tl.max(_qk_0, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_1, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_2, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_3, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_4, axis=0), qk_max)
        _qk_max = tl.maximum(tl.max(_qk_5, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_6, axis=0), _qk_max)
        _qk_max = tl.maximum(tl.max(_qk_7, axis=0), _qk_max)
        exp_tmp = tl.exp(_qk_0 - _qk_max) + tl.exp(_qk_1 - _qk_max) + tl.exp(
            _qk_2 - _qk_max) + tl.exp(_qk_3 - _qk_max) + tl.exp(_qk_4 - _qk_max
            ) + tl.exp(_qk_5 - _qk_max) + tl.exp(_qk_6 - _qk_max) + tl.exp(
            _qk_7 - _qk_max)
        _exp_sum = exp_sum * tl.exp(qk_max - _qk_max) + tl.sum(exp_tmp, axis=0)
        qkv_sum_tmp = tl.exp(_qk_0[:, None] - _qk_max) * v_0 + tl.exp(_qk_1
            [:, None] - _qk_max) * v_1 + tl.exp(_qk_2[:, None] - _qk_max
            ) * v_2 + tl.exp(_qk_3[:, None] - _qk_max) * v_3 + tl.exp(_qk_4
            [:, None] - _qk_max) * v_4 + tl.exp(_qk_5[:, None] - _qk_max
            ) * v_5 + tl.exp(_qk_6[:, None] - _qk_max) * v_6 + tl.exp(_qk_7
            [:, None] - _qk_max) * v_7
        qkv = (qkv * (exp_sum * tl.exp(qk_max - _qk_max)) + qkv_sum_tmp
            ) / _exp_sum
        qk_max = _qk_max
        exp_sum = _exp_sum
    return qkv, qk_max, exp_sum
