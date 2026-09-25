# SPDX-License-Identifier: BSD-3-Clause
"""PyTorch reference for the SmolLM-135M fp16 prefill math.

Weight layout matches the ONNX MatMul initializers: ``[K, N]`` so
``y = x @ weight``. This is the transpose of a Hugging Face ``nn.Linear``
weight.
"""

import math

import torch

from config import SmolConfig


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps)
    return (y * weight.float()).to(x.dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x.float()).to(x.dtype)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return silu(gate) * up


def build_rope_cache(seq_len: int, head_dim: int, theta: float, dtype: torch.dtype):
    """Llama rotary cache. ``cos`` and ``sin`` have shape ``[seq, head_dim]``."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Llama RoPE. ``x`` is ``[heads, seq, dim]``, cache is ``[seq, dim]``."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[None, :, :]
    sin = sin[None, :, :]
    y1 = x1.float() * cos[..., :half].float() - x2.float() * sin[..., :half].float()
    y2 = x2.float() * cos[..., half:].float() + x1.float() * sin[..., half:].float()
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads. ``x`` is ``[kv_heads, seq, dim]``."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=0)


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Causal attention. Tensors are ``[heads, seq, dim]``."""
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    seq = q.shape[-2]
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def project_qkv(hidden: torch.Tensor, weight: torch.Tensor, n_heads: int, head_dim: int):
    seq = hidden.shape[0]
    projected = hidden @ weight
    return projected.view(seq, n_heads, head_dim).permute(1, 0, 2).contiguous()


def merge_heads(attn: torch.Tensor) -> torch.Tensor:
    # [heads, seq, dim] -> [seq, hidden]
    return attn.permute(1, 0, 2).contiguous().view(attn.shape[1], -1)


def decoder_layer(x, layer, cos, sin, cfg: SmolConfig):
    residual = x
    h = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
    q = project_qkv(h, layer.q, cfg.num_attention_heads, cfg.head_dim)
    k = project_qkv(h, layer.k, cfg.num_key_value_heads, cfg.head_dim)
    v = project_qkv(h, layer.v, cfg.num_key_value_heads, cfg.head_dim)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    k = repeat_kv(k, cfg.num_kv_groups)
    v = repeat_kv(v, cfg.num_kv_groups)
    scale = 1.0 / math.sqrt(cfg.head_dim)
    attn = causal_attention(q, k, v, scale)
    x = residual + (merge_heads(attn) @ layer.o)
    residual = x
    h = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
    x = residual + (swiglu(h @ layer.gate, h @ layer.up) @ layer.down)
    return x.to(torch.float16)


def prefill(input_ids: torch.Tensor, weights, num_layers: int | None = None, logits: bool = True):
    """Run a dense causal prefill. ``input_ids`` is a 1-D int64 tensor."""
    cfg: SmolConfig = weights.cfg
    x = weights.embed[input_ids].to(torch.float16).contiguous()
    cos, sin = build_rope_cache(x.shape[0], cfg.head_dim, cfg.rope_theta, torch.float16)
    layers = weights.layers if num_layers is None else weights.layers[:num_layers]
    for layer in layers:
        x = decoder_layer(x, layer, cos, sin, cfg)
    x = rms_norm(x, weights.final_norm, cfg.rms_norm_eps)
    if not logits:
        return x
    # Tied embeddings. embed is [vocab, hidden], so the head is its transpose.
    return (x.float() @ weights.embed.float().t()).to(torch.float16)
