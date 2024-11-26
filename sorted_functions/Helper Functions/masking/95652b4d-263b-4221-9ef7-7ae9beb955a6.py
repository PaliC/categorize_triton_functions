import triton
import triton.language as tl
import torch

@triton.jit
def maximum_path_cp(path, value, t_x, t_y, B, T, S, max_neg_val,
    BLOCK_SIZE_X: 'tl.constexpr'):
    batch = tl.program_id(axis=0)
    path += batch * T * S
    value += batch * T * S
    x_length = tl.load(t_x + batch)
    y_length = tl.load(t_y + batch)
    offs_prev = tl.arange(0, BLOCK_SIZE_X)
    init = tl.where(offs_prev == 0, tl.load(value), max_neg_val)
    tl.store(value + offs_prev * S, init, mask=offs_prev < x_length)
    for j in range(1, y_length, 1):
        v_cur = tl.load(value + offs_prev * S + (j - 1), mask=offs_prev <
            x_length, other=max_neg_val)
        v_prev = tl.load(value + (offs_prev - 1) * S + (j - 1), mask=(0 <
            offs_prev) & (offs_prev < x_length), other=max_neg_val)
        v = tl.maximum(v_cur, v_prev) + tl.load(value + offs_prev * S + j,
            mask=offs_prev < x_length)
        tl.store(value + offs_prev * S + j, v, mask=offs_prev < x_length)
    index = x_length - 1
    for j in range(y_length - 1, -1, -1):
        tl.store(path + index * S + j, 1)
        if index > 0:
            v_left = tl.load(value + index * S + j - 1)
            v_leftdown = tl.load(value + (index - 1) * S + j - 1)
            if v_left < v_leftdown:
                index += -1
