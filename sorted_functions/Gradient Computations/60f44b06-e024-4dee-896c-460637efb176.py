import triton
import triton.language as tl
import torch

@triton.jit
def dequantize(x_, scale, shift, PACKED_PER_VAL: 'tl.constexpr'):
    """PACKED_PER_VAL is the number of values packed into each element x_.
    For example, for int4 quantization and x_ of type int32, PACKED_PER_VAL is 8.
    """
    BLOCK_N: 'tl.constexpr' = x_.shape[0]
    BLOCK_DMODEL_PACKED: 'tl.constexpr' = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * (32 // PACKED_PER_VAL)
    quant_offset = x_[:, :, None, :] >> offsets
    quant_offset = tl.reshape(quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED *
        PACKED_PER_VAL))
    if PACKED_PER_VAL == 4:
        fp8_type = (tl.float8e4b8 if torch.version.hip is not None else tl.
            float8e4nv)
        dequant = quant_offset.to(tl.uint8).to(fp8_type, bitcast=True
            ) * scale + shift
    else:
        quant_offset = (quant_offset & 15).to(tl.uint16)
        quant_offset = quant_offset * 32768.0
        scale_512 = scale * 512
        dequant = quant_offset * scale_512 + shift
    return dequant
