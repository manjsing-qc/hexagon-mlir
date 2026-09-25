# SPDX-License-Identifier: BSD-3-Clause
"""CPU checks for ONNX weight loading and the PyTorch reference prefill."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onnx_weights import default_onnx_path, load_onnx_weights
from reference import prefill


def test_onnx_weight_shapes():
    path = default_onnx_path()
    if path is None:
        raise SystemExit("Set SMOLLM_ONNX to the fp16 SmolLM ONNX file")
    weights = load_onnx_weights(path, num_layers=2)
    cfg = weights.cfg
    assert len(weights.layers) == 2
    layer = weights.layers[0]
    assert tuple(layer.q.shape) == (cfg.hidden_size, cfg.q_out)
    assert tuple(layer.k.shape) == (cfg.hidden_size, cfg.kv_out)
    assert tuple(layer.down.shape) == (cfg.intermediate_size, cfg.hidden_size)
    assert tuple(layer.input_norm.shape) == (cfg.hidden_size,)
    assert weights.embed.dtype == torch.float16


def test_reference_matches_onnxruntime():
    path = default_onnx_path()
    if path is None:
        raise SystemExit("Set SMOLLM_ONNX to the fp16 SmolLM ONNX file")
    import numpy as np
    import onnxruntime as ort

    weights = load_onnx_weights(path)
    # The published ONNX graph is a one-token decode with a KV cache.
    # An empty cache plus one token is the same math as a length-1 prefill.
    token = int(os.environ.get("SMOLLM_TOKEN", "42"))
    ids = torch.tensor([token], dtype=torch.long)
    reference = prefill(ids, weights, logits=True)[0].float()

    past = 0
    feeds = {
        "input_ids": np.array([[token]], dtype=np.int64),
        "attention_mask": np.ones((1, past + 1), dtype=np.int64),
        "position_ids": np.array([[0]], dtype=np.int64),
    }
    for layer in range(weights.cfg.num_hidden_layers):
        feeds[f"past_key_values.{layer}.key"] = np.zeros((1, 3, past, 64), dtype=np.float32)
        feeds[f"past_key_values.{layer}.value"] = np.zeros((1, 3, past, 64), dtype=np.float32)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], feeds)[0][0, 0]
    got = torch.from_numpy(logits)
    diff = (reference - got).abs()
    print(f"onnxruntime max abs diff {diff.max().item():.5f} mean {diff.mean().item():.5f}")
    # fp16 weights through two runtimes. Top token should still agree.
    assert int(reference.argmax()) == int(got.argmax())
    assert diff.max().item() < 1.5


if __name__ == "__main__":
    test_onnx_weight_shapes()
    print("weight shapes ok")
    test_reference_matches_onnxruntime()
    print("onnxruntime match ok")
