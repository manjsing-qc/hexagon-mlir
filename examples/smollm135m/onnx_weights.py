# SPDX-License-Identifier: BSD-3-Clause
"""Load SmolLM-135M fp16 weights from the ONNX decode export.

Projection matrices are anonymous ``onnx::MatMul_*`` initializers. The graph
node names still carry the Hugging Face module path, and each matrix is stored
as ``[K, N]`` for ``MatMul(x, W)``.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from config import SmolConfig

_LAYER_MATMUL = re.compile(
    r"/model/layers\.(\d+)/(self_attn|mlp)/(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)/MatMul"
)
_LAYER_NORM = re.compile(r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight")


@dataclass
class LayerWeights:
    input_norm: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    post_norm: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass
class SmolWeights:
    cfg: SmolConfig
    embed: torch.Tensor
    layers: list[LayerWeights] = field(default_factory=list)
    final_norm: torch.Tensor | None = None


def default_onnx_path() -> Path | None:
    env = os.environ.get("SMOLLM_ONNX")
    candidates = [
        Path(env) if env else None,
        Path("/home/ubuntu/hexagon-artifacts/models/smollm-135m/model_fp16.onnx"),
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path
    return None


def _to_fp16(array) -> torch.Tensor:
    tensor = torch.from_numpy(array.copy())
    if tensor.dtype != torch.float16:
        tensor = tensor.to(torch.float16)
    return tensor.contiguous()


def load_onnx_weights(path: str | Path, num_layers: int | None = None) -> SmolWeights:
    import onnx
    from onnx import numpy_helper

    cfg = SmolConfig()
    model = onnx.load(str(path), load_external_data=True)
    initializers = {item.name: item for item in model.graph.initializer}

    def take(name: str) -> torch.Tensor:
        if name not in initializers:
            raise KeyError(f"ONNX initializer {name} not found")
        return _to_fp16(numpy_helper.to_array(initializers[name]))

    embed = take("model.embed_tokens.weight")
    final_norm = take("model.norm.weight")
    if embed.shape != (cfg.vocab_size, cfg.hidden_size):
        raise ValueError(f"unexpected embedding shape {tuple(embed.shape)}")
    if final_norm.shape != (cfg.hidden_size,):
        raise ValueError(f"unexpected final norm shape {tuple(final_norm.shape)}")

    limit = cfg.num_hidden_layers if num_layers is None else num_layers
    layers: list[dict] = [{} for _ in range(limit)]
    for name in initializers:
        match = _LAYER_NORM.match(name)
        if not match:
            continue
        index = int(match.group(1))
        if index >= limit:
            continue
        kind = match.group(2)
        layers[index]["input_norm" if kind == "input_layernorm" else "post_norm"] = take(name)

    expected = {
        "q": (cfg.hidden_size, cfg.q_out),
        "k": (cfg.hidden_size, cfg.kv_out),
        "v": (cfg.hidden_size, cfg.kv_out),
        "o": (cfg.hidden_size, cfg.hidden_size),
        "gate": (cfg.hidden_size, cfg.intermediate_size),
        "up": (cfg.hidden_size, cfg.intermediate_size),
        "down": (cfg.intermediate_size, cfg.hidden_size),
    }
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        match = _LAYER_MATMUL.match(node.name)
        if not match:
            continue
        index = int(match.group(1))
        if index >= limit:
            continue
        proj = {
            "q_proj": "q",
            "k_proj": "k",
            "v_proj": "v",
            "o_proj": "o",
            "gate_proj": "gate",
            "up_proj": "up",
            "down_proj": "down",
        }[match.group(3)]
        weight_name = next(
            (item for item in node.input if item in initializers and len(initializers[item].dims) == 2),
            None,
        )
        if weight_name is None:
            raise ValueError(f"{node.name} has no rank-2 initializer")
        weight = take(weight_name)
        if tuple(weight.shape) != expected[proj]:
            raise ValueError(f"{node.name} has shape {tuple(weight.shape)}, expected {expected[proj]}")
        layers[index][proj] = weight

    built = []
    fields = ("input_norm", "q", "k", "v", "o", "post_norm", "gate", "up", "down")
    for index, layer in enumerate(layers):
        missing = [name for name in fields if name not in layer]
        if missing:
            raise ValueError(f"layer {index} is missing {missing}")
        built.append(LayerWeights(**layer))

    return SmolWeights(cfg=cfg, embed=embed, layers=built, final_norm=final_norm)
