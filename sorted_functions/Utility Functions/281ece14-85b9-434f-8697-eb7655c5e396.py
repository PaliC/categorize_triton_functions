import triton
import triton.language as tl
import torch

@triton.jit
def print_tensor_dim(tensor, str_name):
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tl.static_print(str_name, ' ', tensor.shape, ' ', tensor.dtype)