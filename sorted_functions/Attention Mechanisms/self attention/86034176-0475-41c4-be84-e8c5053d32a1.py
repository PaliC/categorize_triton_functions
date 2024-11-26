import triton
import triton.language as tl
import torch

@triton.jit
def _single_query_cached_kv_attention_v2(exp_sums, max_logits, out, q,
    k_cache, v_cache, head_mapping, scale, block_tables, seq_lens,
    partiton_size, max_num_blocks_per_seq, alibi_slopes, stride_qm,
    stride_qn, stride_om, stride_on, stride_ok, stride_km, stride_kn,
    stride_kk, stride_exp_m, stride_exp_n, BLOCK_SIZE: 'tl.constexpr',
    HEAD_SIZE: 'tl.constexpr'):
    seq_idx = tl.program_id(axis=1)
    par_idx = tl.program_id(axis=2)
    seq_len = tl.load(seq_lens + seq_idx)
    if par_idx * partiton_size >= seq_len:
        return
    num_context_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    num_blocks_per_par = partiton_size // BLOCK_SIZE
    start_block_idx = par_idx * num_blocks_per_par
    end_block_idx = tl.minimum(start_block_idx + num_blocks_per_par,
        num_context_blocks)
    head_idx = tl.program_id(axis=0)
    kv_head_idx = tl.load(head_mapping + head_idx)
    if alibi_slopes is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes + head_idx)
    block_offs = tl.arange(0, BLOCK_SIZE)
    head_size_offs = tl.arange(0, HEAD_SIZE)
    q = tl.load(q + seq_idx * stride_qm + head_idx * stride_qn + head_size_offs
        )
    q = q * scale
    qkv = tl.zeros([BLOCK_SIZE, HEAD_SIZE], dtype=tl.float32)
    qk_max = float('-inf')
    exp_sum = 0.0
    fp16_0 = tl.zeros([1, 1], dtype=k_cache.dtype.element_ty)
    base_offs_kv = kv_head_idx * stride_kn + block_offs[:, None
        ] * stride_kk + head_size_offs[None, :]
    for block_idx in range(start_block_idx, end_block_idx):
        physical_block_idx = tl.load(block_tables + seq_idx *
            max_num_blocks_per_seq + block_idx)
        mask = (block_offs[:, None] < seq_len - block_idx * BLOCK_SIZE) & (
            head_size_offs[None, :] < HEAD_SIZE)
        offs_kv = physical_block_idx * stride_km + base_offs_kv
        k = tl.load(k_cache + offs_kv, mask=mask, other=fp16_0)
        v = tl.load(v_cache + offs_kv, mask=mask, other=fp16_0)
        _qk = tl.sum(q[None, :] * k, axis=1)
        _qk += alibi_slope * (block_idx * BLOCK_SIZE + block_offs - seq_len + 1
            )
        _qk_max = tl.maximum(tl.max(_qk, axis=0), qk_max)
        qk = tl.where(block_offs[:, None] < seq_len - block_idx *
            BLOCK_SIZE, _qk[:, None], float('-inf'))
        _exp_sum = exp_sum * tl.exp(qk_max - _qk_max) + tl.sum(tl.exp(_qk -
            _qk_max), axis=0)
        qkv = qkv * (exp_sum * tl.exp(qk_max - _qk_max) / _exp_sum) + tl.exp(
            qk - _qk_max) / _exp_sum * v
        qk_max = _qk_max
        exp_sum = _exp_sum
    offs_exp = seq_idx * stride_exp_m + head_idx * stride_exp_n + par_idx
    tl.store(exp_sums + offs_exp, exp_sum)
    tl.store(max_logits + offs_exp, qk_max)
    offs_out = (seq_idx * stride_om + head_idx * stride_on + par_idx *
        stride_ok + head_size_offs)
    tl.store(out + offs_out, tl.sum(qkv, axis=0))
