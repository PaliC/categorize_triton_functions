import triton
import triton.language as tl
import torch

@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,
    'BLOCK_DMODEL': 64}, num_stages=3, num_warps=4)], key=['num_queries'])
@triton.jit
def triton_tem_fused_with_exp2(arg_Q, arg_K, arg_V, out_ptr0, num_queries:
    'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr',
    BLOCK_DMODEL: 'tl.constexpr'):
    SCORE_MOD_IS_LINEAR: 'tl.constexpr' = False
    ROWS_GUARANTEED_SAFE: 'tl.constexpr' = False
    Q = arg_Q
    K = arg_K
    V = arg_V
    stride_qz = 4194304
    stride_qh = 262144
    stride_qm = 64
    stride_qk = 1
    stride_kz = 4194304
    stride_kh = 262144
    stride_kn = 64
    stride_kk = 1
    stride_vz = 4194304
    stride_vh = 262144
    stride_vk = 64
    stride_vn = 1
    Z = 16
    H = 16
    N_CTX = 4096
    qk_scale = 1.0
    MATMUL_PRECISION = Q.dtype.element_ty
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qkv_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(base=Q + qkv_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m *
        BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qkv_offset, shape=(
        BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0
        ), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qkv_offset, shape=(N_CTX,
        BLOCK_DMODEL), strides=(stride_vk, stride_vn), offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q = tl.load(Q_block_ptr)
    if SCORE_MOD_IS_LINEAR:
        qk_scale *= 1.44269504
    q = q * qk_scale
    lo = 0
    hi = N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk)
        tmp0 = tl.full([1], 1024, tl.int64)
        tmp1 = offs_m[:, None] <= tmp0
        tmp2 = start_n + offs_n[None, :] <= tmp0
        tmp3 = tmp1 & tmp2
        tmp4 = offs_m[:, None] >= start_n + offs_n[None, :]
        tmp5 = tmp3 | tmp4
        tmp6 = float('-inf')
        tmp7 = tmp6
        tmp8 = tl.where(tmp5, qk, tmp7)
        qk = tmp8
        if not SCORE_MOD_IS_LINEAR:
            qk *= 1.44269504
        row_max = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, row_max)
        masked_out_rows = m_i_new == float('-inf')
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        if not ROWS_GUARANTEED_SAFE:
            alpha = tl.where(masked_out_rows, 0, alpha)
            p = tl.where(masked_out_rows[:, None], 0, p)
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    acc = acc / l_i[:, None]
    idx_z = tl.program_id(1) // H
    idx_h = tl.program_id(1) % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]
    mask = (idx_m != -1) & (idx_d != -1)
    xindex = idx_d + 64 * idx_m + 262144 * idx_h + 4194304 * idx_z
    tl.store(out_ptr0 + xindex, acc, None)
