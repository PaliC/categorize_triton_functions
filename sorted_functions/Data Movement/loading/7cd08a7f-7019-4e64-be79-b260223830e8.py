import triton
import triton.language as tl
import torch

@triton.jit
def backward_scan(gates, tokens, outputs, SEQUENCE_LENGTH: 'tl.constexpr'):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0
        ) + tl.program_id(axis=1)
    forward_strides = tl.arange(0, SEQUENCE_LENGTH
        ) + sequence_id * SEQUENCE_LENGTH
    reverse_strides = tl.num_programs(axis=0) * tl.num_programs(axis=1
        ) * SEQUENCE_LENGTH - 1 - forward_strides
    tokens_ = tl.load(tokens + reverse_strides)
    gates_ = tl.load(gates + reverse_strides)
    tuples = pack64(tokens_, gates_)
    output_tuples_ = tl.associative_scan(tuples, axis=0, combine_fn=
        first_order_op)
    output_tokens_, output_gates_ = unpack64(output_tuples_)
    tl.store(outputs + reverse_strides, output_tokens_)
