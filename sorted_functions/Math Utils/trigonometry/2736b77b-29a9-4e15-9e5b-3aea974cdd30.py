import triton
import triton.language as tl
import torch

@triton.jit
def _triton_first_order_fwd(x_ptr: 'tl.tensor', y_ptr: 'tl.tensor', z_ptr:
    'tl.tensor', sph_1_0_ptr: 'tl.tensor', sph_1_1_ptr: 'tl.tensor',
    sph_1_2_ptr: 'tl.tensor', BLOCK_SIZE: 'tl.constexpr', vector_length:
    'tl.constexpr'):
    """
    First order spherical harmonics in Triton.

    Computationally not that intensive, as we're just applying
    a sqrt 3 to the coordinates, but also good for validating
    the kernel performs as intended.

    Parameters
    ----------
    x_ptr, y_ptr, z_ptr : tl.tensor
        Pointers to the coordinate tensors.
    sph_1_0_ptr, sph_1_1_ptr, sph_1_2_ptr : tl.tensor
        Points to tensors to write outputs to. Assumed to
        be the same length as the input tensors.
    block_size : tl.constexpr
        Vector length of contiguous elements to load into memory
        within a given block.
    vector_length : tl.constexpr
        The maximum/total length of the vectors, assumed to
        be the same for every one. This is used to calculate
        the mask to keep operations within bounds.
    """
    sqrt_3 = 3 ** 0.5
    block_id = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * block_id
    x_row_start = x_ptr + offset
    y_row_start = y_ptr + offset
    z_row_start = z_ptr + offset
    x = tl.load(x_row_start, mask=offset < vector_length)
    y = tl.load(y_row_start, mask=offset < vector_length)
    z = tl.load(z_row_start, mask=offset < vector_length)
    sph_1_0 = sqrt_3 * x
    sph_1_1 = sqrt_3 * y
    sph_1_2 = sqrt_3 * z
    sph_1_0_start = sph_1_0_ptr + offset
    sph_1_1_start = sph_1_1_ptr + offset
    sph_1_2_start = sph_1_2_ptr + offset
    tl.store(sph_1_0_start, sph_1_0, mask=offset < vector_length)
    tl.store(sph_1_1_start, sph_1_1, mask=offset < vector_length)
    tl.store(sph_1_2_start, sph_1_2, mask=offset < vector_length)
