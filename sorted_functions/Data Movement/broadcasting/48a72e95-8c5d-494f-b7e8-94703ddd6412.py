import triton
import triton.language as tl
import torch

@triton.jit
def triton_unbroadcast(array, other):
    l: 'tl.constexpr' = tl.constexpr(shape_l(array.shape))
    ol: 'tl.constexpr' = tl.constexpr(shape_l(other.value))
    for i in tl.static_range(0, l):
        if i >= ol:
            array = tl.sum(array, l - (1 + i))
            array = tl.expand_dims(array, l - (1 + i))
        elif array.shape[l - (1 + i)] > other.value[ol - (1 + i)]:
            array = tl.sum(array, l - (1 + i))
            array = tl.expand_dims(array, l - (1 + i))
        tl.static_assert(tl.constexpr(shape_l(array.shape)) == l)
    return tl.view(array, other.value)
