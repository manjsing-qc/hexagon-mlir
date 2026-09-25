# SPDX-License-Identifier: BSD-3-Clause
"""Simulator checks for the SmolLM fp16 Triton kernels.

Run one case at a time. Each launch compiles and executes on hexagon-sim:

    python test_kernels.py residual
    python test_kernels.py linear
    python test_kernels.py attention
    python test_kernels.py layer
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SmolConfig
from kernels import causal_flash_attention, linear, residual, rms_norm, rope, swiglu
from reference import apply_rope, build_rope_cache, causal_attention, decoder_layer, swiglu as swiglu_ref
from reference import rms_norm as rms_norm_ref
from onnx_weights import LayerWeights


def _report(name, got, expected):
    diff = (got.float() - expected.float()).abs()
    print(f"{name}: max abs {diff.max().item():.5f} mean {diff.mean().item():.5f}")
    return diff.max().item()


def test_residual():
    torch.manual_seed(0)
    a = torch.randn(16, 576, dtype=torch.float16)
    b = torch.randn(16, 576, dtype=torch.float16)
    got = residual(a, b)
    err = _report("residual", got, a + b)
    assert err < 1e-2


def test_rmsnorm():
    torch.manual_seed(0)
    x = torch.randn(16, 576, dtype=torch.float16)
    weight = torch.randn(576, dtype=torch.float16)
    got = rms_norm(x, weight, 1e-5)
    err = _report("rmsnorm", got, rms_norm_ref(x, weight, 1e-5))
    assert err < 2e-2


def test_swiglu():
    torch.manual_seed(0)
    gate = torch.randn(16, 1536, dtype=torch.float16)
    up = torch.randn(16, 1536, dtype=torch.float16)
    got = swiglu(gate, up)
    err = _report("swiglu", got, swiglu_ref(gate, up))
    assert err < 2e-2


def test_linear():
    torch.manual_seed(0)
    x = (torch.randn(16, 64, dtype=torch.float16) * 0.1)
    w = (torch.randn(64, 64, dtype=torch.float16) * 0.1)
    got = linear(x, w)
    err = _report("linear-64", got, x @ w)
    assert err < 2e-1


def test_linear_smol_tile():
    torch.manual_seed(0)
    x = torch.randn(16, 576, dtype=torch.float16) * 0.05
    w = torch.randn(576, 64, dtype=torch.float16) * 0.05
    got = linear(x, w)
    err = _report("linear-576x64", got, x @ w)
    assert err < 3e-1


def test_rope():
    torch.manual_seed(0)
    seq, heads, dim = 16, 9, 64
    x = torch.randn(heads, seq, dim, dtype=torch.float16)
    cos, sin = build_rope_cache(seq, dim, 10000.0, torch.float16)
    got = rope(x, cos, sin)
    err = _report("rope", got, apply_rope(x, cos, sin))
    assert err < 2e-2


def test_attention():
    torch.manual_seed(0)
    heads, seq, dim = 1, 32, 64
    q = torch.randn(heads, seq, dim, dtype=torch.float16) * 0.1
    k = torch.randn(heads, seq, dim, dtype=torch.float16) * 0.1
    v = torch.randn(heads, seq, dim, dtype=torch.float16) * 0.1
    got = causal_flash_attention(q, k, v, block_n=16)
    expected = causal_attention(q, k, v, scale=1.0 / (dim**0.5))
    err = _report("causal-fa2", got, expected)
    assert err < 2e-1


def test_layer():
    """One SmolLM-shaped decoder layer on random fp16 weights."""
    torch.manual_seed(0)
    cfg = SmolConfig()
    seq = 16

    def param(*shape):
        return torch.randn(*shape, dtype=torch.float16) * 0.02

    layer = LayerWeights(
        input_norm=param(cfg.hidden_size),
        q=param(cfg.hidden_size, cfg.q_out),
        k=param(cfg.hidden_size, cfg.kv_out),
        v=param(cfg.hidden_size, cfg.kv_out),
        o=param(cfg.hidden_size, cfg.hidden_size),
        post_norm=param(cfg.hidden_size),
        gate=param(cfg.hidden_size, cfg.intermediate_size),
        up=param(cfg.hidden_size, cfg.intermediate_size),
        down=param(cfg.intermediate_size, cfg.hidden_size),
    )
    x = torch.randn(seq, cfg.hidden_size, dtype=torch.float16) * 0.02
    cos, sin = build_rope_cache(seq, cfg.head_dim, cfg.rope_theta, torch.float16)
    from pipeline import decoder_layer as hexagon_layer

    got = hexagon_layer(x, layer, cos, sin, cfg)
    expected = decoder_layer(x, layer, cos, sin, cfg)
    err = _report("decoder-layer", got, expected)
    assert err < 0.5


CASES = {
    "residual": test_residual,
    "rmsnorm": test_rmsnorm,
    "swiglu": test_swiglu,
    "linear": test_linear,
    "linear-smol": test_linear_smol_tile,
    "rope": test_rope,
    "attention": test_attention,
    "layer": test_layer,
}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "residual"
    if name == "all-kernels":
        for key in ("residual", "rmsnorm", "swiglu", "linear", "linear-smol", "rope", "attention"):
            CASES[key]()
    else:
        CASES[name]()
    print(f"passed: {name}")
