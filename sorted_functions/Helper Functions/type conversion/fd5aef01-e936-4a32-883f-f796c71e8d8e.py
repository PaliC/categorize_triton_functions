import triton
import triton.language as tl
import torch

@triton.jit
def hello_triton():
    tl.device_print('Hello Triton!')
