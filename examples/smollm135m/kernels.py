# SPDX-License-Identifier: BSD-3-Clause
"""fp16 Triton kernels for SmolLM-135M, lowered through the Hexagon backend.

Every launch goes through Triton -> triton-shared -> linalg-hexagon -> the
Hexagon simulator or device. Shapes are static. Sequence length and tile sizes
must be powers of two because ``tl.arange`` and ``tl.dot`` require that.
"""

import math

import torch
import triton
import triton.language as tl


def activate():
    from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

    triton.runtime.driver.set_active(HexagonDriver())


# Flags that match the kernels already known to run on hexagon-sim.
_ELEMENTWISE = dict(
    enableMultiThreading=False,
    enableVTCMTiling=True,
    enableConvertToHexagonmem=True,
    enableHexagonmemCopyToDMA=True,
    enableVectorization=True,
)
_MATMUL = dict(
    enableMultiThreading=False,
    enableVTCMTiling=True,
    enableConvertToHexagonmem=True,
    enableHexagonmemCopyToDMA=True,
    enableVectorization=True,
    enableHexKL=False,
)
# Flash attention in this repo is lowered with VTCM tiling and DMA copies off.
_ATTENTION = dict(
    enableVectorization=True,
    enableSplitReduceGeneric=True,
    enableHVXInlining=True,
    enableSCFLoopUnroll=False,
    enableMultiThreading=False,
    enableHexKL=False,
    enableVTCMTiling=False,
    enableConvertToHexagonmem=False,
    enableHexagonmemCopyToDMA=False,
)


@triton.jit
def residual_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + offs, mask=mask, other=0)
        b = tl.load(b_ptr + offs, mask=mask, other=0)
        tl.store(out_ptr + offs, a + b, mask=mask)


@triton.jit
def rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    mask = cols < COLS
    weight = tl.load(w_ptr + cols, mask=mask, other=0).to(tl.float32)
    for row in range(0, ROWS):
        x = tl.load(x_ptr + row * COLS + cols, mask=mask, other=0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / COLS
        y = (x * tl.rsqrt(mean_sq + EPS)) * weight
        tl.store(y_ptr + row * COLS + cols, y.to(tl.float16), mask=mask)


@triton.jit
def swiglu_kernel(gate_ptr, up_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        gate = tl.load(gate_ptr + offs, mask=mask, other=0)
        up = tl.load(up_ptr + offs, mask=mask, other=0)
        # SiLU in fp32, then the SwiGLU product. Matches the Hexagon SiLU test.
        silu = gate / (1 + tl.exp((-gate).to(tl.float32)).to(gate.dtype))
        tl.store(out_ptr + offs, silu * up, mask=mask)


@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    HEADS: tl.constexpr,
    SEQ: tl.constexpr,
    DIM: tl.constexpr,
    HALF: tl.constexpr,
):
    """Llama RoPE. ``x`` is ``[heads, seq, dim]``; the cache is ``[seq, dim]``.

    Each half of the head dimension is a 2D block pointer. triton-shared
    rejects the flattened modulo addressing that a single 1D loop would use.
    """
    for head in range(HEADS):
        x_base = x_ptr + head * SEQ * DIM
        y_base = out_ptr + head * SEQ * DIM
        x1_ptr = tl.make_block_ptr(
            base=x_base,
            shape=(SEQ, DIM),
            strides=(DIM, 1),
            offsets=(0, 0),
            block_shape=(SEQ, HALF),
            order=(1, 0),
        )
        x2_ptr = tl.make_block_ptr(
            base=x_base,
            shape=(SEQ, DIM),
            strides=(DIM, 1),
            offsets=(0, HALF),
            block_shape=(SEQ, HALF),
            order=(1, 0),
        )
        c1_ptr = tl.make_block_ptr(
            base=cos_ptr,
            shape=(SEQ, DIM),
            strides=(DIM, 1),
            offsets=(0, 0),
            block_shape=(SEQ, HALF),
            order=(1, 0),
        )
        c2_ptr = tl.make_block_ptr(
            base=cos_ptr,
            shape=(SEQ, DIM),
            strides=(DIM, 1),
            offsets=(0, HALF),
            block_shape=(SEQ, HALF),
            order=(1, 0),
        )
        s1_ptr = tl.make_block_ptr(
            base=sin_ptr,
            shape=(SEQ, DIM),
            strides=(DIM, 1),
            offsets=(0, 0),
            block_shape=(SEQ, HALF),
            order=(1, 0),
        )
        s2_ptr = tl.make_block_ptr(
            base=sin_ptr,
            shape=(SEQ, DIM),
            strides=(DIM, 1),
            offsets=(0, HALF),
            block_shape=(SEQ, HALF),
            order=(1, 0),
        )
        y1_ptr = tl.make_block_ptr(
            base=y_base,
            shape=(SEQ, DIM),
            strides=(DIM, 1),
            offsets=(0, 0),
            block_shape=(SEQ, HALF),
            order=(1, 0),
        )
        y2_ptr = tl.make_block_ptr(
            base=y_base,
            shape=(SEQ, DIM),
            strides=(DIM, 1),
            offsets=(0, HALF),
            block_shape=(SEQ, HALF),
            order=(1, 0),
        )
        x1 = tl.load(x1_ptr).to(tl.float32)
        x2 = tl.load(x2_ptr).to(tl.float32)
        c1 = tl.load(c1_ptr).to(tl.float32)
        c2 = tl.load(c2_ptr).to(tl.float32)
        s1 = tl.load(s1_ptr).to(tl.float32)
        s2 = tl.load(s2_ptr).to(tl.float32)
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
        tl.store(y1_ptr, y1.to(tl.float16))
        tl.store(y2_ptr, y2.to(tl.float16))


@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Full ``y = x @ w`` in one launch.

    Column tiles stay inside the kernel so a wide matmul, including the tied
    LM head, does not pay a simulator boot per tile. ``BLOCK_N`` must divide N.
    """
    offs_m = tl.arange(0, M)[:, None]
    offs_bn = tl.arange(0, BLOCK_N)[None, :]
    for n0 in range(0, N, BLOCK_N):
        acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + offs_m * K + offs_k[None, :])
            w = tl.load(w_ptr + offs_k[:, None] * N + (n0 + offs_bn))
            acc = tl.dot(x, w, acc)
        tl.store(y_ptr + offs_m * N + (n0 + offs_bn), acc.to(tl.float16))


@triton.jit
def linear_tile_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    n0,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One ``BLOCK_N`` column tile of ``y = x @ w`` with an fp32 accumulator.

    ``x`` is ``[M, K]``, ``w`` is ``[K, N]``, both fp16 and row-major.
    ``K`` is reduced in ``BLOCK_K`` steps so each ``tl.dot`` stays on a
    power-of-two inner dimension.
    """
    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    offs_m = tl.arange(0, M)[:, None]
    offs_bn = tl.arange(0, BLOCK_N)[None, :]
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m * K + offs_k[None, :])
        w = tl.load(w_ptr + offs_k[:, None] * N + (n0 + offs_bn))
        acc = tl.dot(x, w, acc)
    tl.store(y_ptr + offs_m * N + (n0 + offs_bn), acc.to(tl.float16))


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    qk_scale,
    offs_m,
    BLOCK_N: tl.constexpr,
    N_CTX: tl.constexpr,
):
    # Online-softmax FlashAttention-2 forward. Each step updates the running
    # row max and sum, so the score matrix is never stored.
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        # Explicit transpose: block-pointer transpose creation fails in this backend.
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal, qk, -1.0e4)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def causal_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    sm_scale: tl.constexpr,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(BLOCK_DMODEL, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(BLOCK_DMODEL, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(BLOCK_DMODEL, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(BLOCK_DMODEL, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    offs_m = tl.arange(0, BLOCK_M)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1.0e10
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    q = tl.load(q_block_ptr)
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_block_ptr,
        v_block_ptr,
        qk_scale,
        offs_m,
        BLOCK_N,
        N_CTX,
    )
    acc = acc / l_i[:, None]
    tl.store(o_block_ptr, acc.to(tl.float16))


def _check_fp16(*tensors):
    for tensor in tensors:
        if tensor.dtype != torch.float16 or not tensor.is_contiguous():
            raise ValueError("Hexagon kernels expect contiguous fp16 tensors")


def residual(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    activate()
    _check_fp16(a, b)
    if a.shape != b.shape:
        raise ValueError(f"residual shape mismatch {a.shape} vs {b.shape}")
    out = torch.empty_like(a)
    n = a.numel()
    block = 1024
    residual_kernel[(1,)](a, b, out, N=n, BLOCK=block, **_ELEMENTWISE)
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    activate()
    _check_fp16(x, weight)
    rows, cols = x.shape
    block = triton.next_power_of_2(cols)
    out = torch.empty_like(x)
    rms_norm_kernel[(1,)](
        x,
        weight,
        out,
        ROWS=rows,
        COLS=cols,
        BLOCK=block,
        EPS=eps,
        **_ELEMENTWISE,
    )
    return out


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    activate()
    _check_fp16(gate, up)
    out = torch.empty_like(gate)
    swiglu_kernel[(1,)](gate, up, out, N=gate.numel(), BLOCK=1024, **_ELEMENTWISE)
    return out


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` is ``[heads, seq, dim]``. ``cos`` and ``sin`` are ``[seq, dim]``."""
    activate()
    _check_fp16(x, cos, sin)
    heads, seq, dim = x.shape
    out = torch.empty_like(x)
    rope_kernel[(1,)](
        x,
        cos,
        sin,
        out,
        HEADS=heads,
        SEQ=seq,
        DIM=dim,
        HALF=dim // 2,
        **_ELEMENTWISE,
    )
    return out


def _dividing_block(n: int) -> int:
    for block in (512, 256, 128, 64, 32, 16):
        if n % block == 0:
            return block
    raise ValueError(f"N={n} cannot be covered by a power-of-two block that divides it")


def matmul(x: torch.Tensor, weight: torch.Tensor, block_k: int = 64) -> torch.Tensor:
    """``x @ weight`` in a single Hexagon launch.

    ``x`` is ``[M, K]`` and ``weight`` is ``[K, N]``, both contiguous fp16.
    M must be a power of two (1 is allowed). N must be divisible by a
    power-of-two column tile.
    """
    activate()
    _check_fp16(x, weight)
    m, k = x.shape
    k_w, n = weight.shape
    if k != k_w:
        raise ValueError(f"linear inner dim {k} != {k_w}")
    if m <= 0 or triton.next_power_of_2(m) != m:
        raise ValueError(f"M={m} must be a positive power of two")
    if k % block_k:
        raise ValueError(f"K={k} must be divisible by BLOCK_K={block_k}")
    block_n = _dividing_block(n)
    out = torch.empty((m, n), dtype=torch.float16)
    linear_kernel[(1,)](
        x,
        weight,
        out,
        M=m,
        K=k,
        N=n,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        **_MATMUL,
    )
    return out


def _column_tiles(n: int):
    """Split N into power-of-two column tiles, largest first.

    ``tl.arange`` / ``tl.dot`` need a power-of-two block, and SmolLM's
    projection widths (192, 576, 1536) are not themselves powers of two.
    """
    tiles = []
    cursor = 0
    for block in (512, 256, 128, 64, 32, 16):
        while cursor + block <= n:
            tiles.append((cursor, block))
            cursor += block
    if cursor != n:
        raise ValueError(f"N={n} cannot be tiled by power-of-two blocks down to 16")
    return tiles


def linear(x: torch.Tensor, weight: torch.Tensor, block_k: int = 64) -> torch.Tensor:
    """``x @ weight`` for fp16 ``x [M, K]`` and ``weight [K, N]``."""
    activate()
    _check_fp16(x, weight)
    m, k = x.shape
    k_w, n = weight.shape
    if k != k_w:
        raise ValueError(f"linear inner dim {k} != {k_w}")
    if m <= 0 or triton.next_power_of_2(m) != m:
        raise ValueError(f"M={m} must be a positive power of two")
    if k % block_k:
        raise ValueError(f"K={k} must be divisible by BLOCK_K={block_k}")
    out = torch.empty((m, n), dtype=torch.float16)
    for n0, block_n in _column_tiles(n):
        linear_tile_kernel[(1,)](
            x,
            weight,
            out,
            n0,
            M=m,
            K=k,
            N=n,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            **_MATMUL,
        )
    return out


def causal_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_n: int = 16) -> torch.Tensor:
    """Causal FlashAttention-2 forward. Inputs are ``[heads, seq, dim]`` fp16.

    One head is one launch so the kernel stays a single query block, which is
    the shape the existing Hexagon attention lowering accepts. KV heads must
    already be repeated to match the query heads.
    """
    activate()
    _check_fp16(q, k, v)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"attention shape mismatch {q.shape} {k.shape} {v.shape}")
    heads, seq, dim = q.shape
    if seq != triton.next_power_of_2(seq) or dim != triton.next_power_of_2(dim):
        raise ValueError(f"seq={seq} and dim={dim} must be powers of two")
    if seq % block_n:
        raise ValueError(f"seq={seq} must be divisible by BLOCK_N={block_n}")
    scale = 1.0 / math.sqrt(dim)
    out = torch.empty_like(q)
    for head in range(heads):
        causal_attention_kernel[(1,)](
            q[head],
            k[head],
            v[head],
            out[head],
            sm_scale=scale,
            N_CTX=seq,
            BLOCK_M=seq,
            BLOCK_DMODEL=dim,
            BLOCK_N=block_n,
            **_ATTENTION,
        )
    return out
