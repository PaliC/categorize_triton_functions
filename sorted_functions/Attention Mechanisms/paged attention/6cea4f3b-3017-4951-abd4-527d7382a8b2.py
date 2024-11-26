import triton
import triton.language as tl
import torch

@triton.jit
def _paged_attention_v2_reduce(out, exp_sums, max_logits, tmp_out,
    context_lens, stride_exp_m, stride_exp_n, stride_out_m, stride_out_n,
    stride_tmp_m, stride_tmp_n, stride_tmp_k, HEAD_SIZE: 'tl.constexpr',
    NUM_PARTITIONS: 'tl.constexpr'):
    seq_idx = tl.program_id(axis=1)
    head_idx = tl.program_id(axis=0)
    context_len = tl.load(context_lens + seq_idx)
    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)
    exp_sum = 0.0
    max_logit = float('-inf')
    offs_logit = seq_idx * stride_exp_m + head_idx * stride_exp_n
    head_size_offs = tl.arange(0, HEAD_SIZE)
    tmp_out_ptr = seq_idx * stride_tmp_m + head_idx * stride_tmp_n
    out_ptr = seq_idx * stride_out_m + head_idx * stride_out_n + head_size_offs
    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)
    global_exp_sum = tl.zeros([1], dtype=tl.float32)
    logits = tl.load(max_logits + offs_logit + tl.arange(0, NUM_PARTITIONS),
        mask=tl.arange(0, NUM_PARTITIONS) < num_partitions, other=float('-inf')
        )
    max_logit = tl.max(logits, axis=0)
    exp_sum = tl.load(exp_sums + offs_logit + tl.arange(0, NUM_PARTITIONS),
        mask=tl.arange(0, NUM_PARTITIONS) < num_partitions, other=0.0)
    rescaled_exp_sum = exp_sum * tl.exp(logits - max_logit)
    global_exp_sum += tl.sum(rescaled_exp_sum, axis=0)
    tmp = tl.load(tmp_out + tmp_out_ptr + tl.arange(0, NUM_PARTITIONS)[:,
        None] * stride_tmp_k + head_size_offs)
    acc += tl.sum(tmp * rescaled_exp_sum[:, None], axis=0)
    inv_sum = 1.0 / (global_exp_sum + 1e-06)
    tl.store(out + out_ptr, acc * inv_sum)
