# SPDX-License-Identifier: BSD-3-Clause
"""Host schedule for a SmolLM-135M fp16 prefill on Hexagon.

The ONNX file is a source of weights. hexagon-mlir does not import the ONNX
graph. This module runs each math op as a Triton kernel and keeps the layer
loop, embedding gather, and grouped-query head repeat on the host.
"""

import torch

from config import SmolConfig
from kernels import causal_flash_attention, linear, residual, rms_norm, rope, swiglu
from reference import build_rope_cache, repeat_kv


def _qkv(hidden, weight, n_heads, head_dim):
    seq = hidden.shape[0]
    return linear(hidden, weight).view(seq, n_heads, head_dim).permute(1, 0, 2).contiguous()


def _merge(attn: torch.Tensor) -> torch.Tensor:
    return attn.permute(1, 0, 2).contiguous().view(attn.shape[1], -1)


def decoder_layer(x: torch.Tensor, layer, cos, sin, cfg: SmolConfig) -> torch.Tensor:
    hidden = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
    query = rope(_qkv(hidden, layer.q, cfg.num_attention_heads, cfg.head_dim), cos, sin)
    key = rope(_qkv(hidden, layer.k, cfg.num_key_value_heads, cfg.head_dim), cos, sin)
    value = _qkv(hidden, layer.v, cfg.num_key_value_heads, cfg.head_dim)
    key = repeat_kv(key, cfg.num_kv_groups)
    value = repeat_kv(value, cfg.num_kv_groups)
    attn = causal_flash_attention(query, key, value)
    x = residual(x, linear(_merge(attn), layer.o))
    hidden = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
    fused = swiglu(linear(hidden, layer.gate), linear(hidden, layer.up))
    return residual(x, linear(fused, layer.down))


def prefill(input_ids: torch.Tensor, weights, num_layers: int | None = None, logits: bool = False):
    """Static-shape causal prefill.

    ``input_ids`` is a 1-D token vector whose length is a power of two.
    Logits are optional: the tied head is a ``[seq, 576] @ [576, 49152]``
    matmul and is much heavier on the simulator than one decoder layer.
    """
    cfg: SmolConfig = weights.cfg
    if input_ids.ndim != 1:
        raise ValueError("prefill expects a 1-D token tensor")
    x = weights.embed[input_ids.to(torch.long)].to(torch.float16).contiguous()
    cos, sin = build_rope_cache(x.shape[0], cfg.head_dim, cfg.rope_theta, torch.float16)
    count = cfg.num_hidden_layers if num_layers is None else num_layers
    for layer in weights.layers[:count]:
        x = decoder_layer(x, layer, cos, sin, cfg)
    x = rms_norm(x, weights.final_norm, cfg.rms_norm_eps)
    if not logits:
        return x
    return linear(x, weights.embed.transpose(0, 1).contiguous())
