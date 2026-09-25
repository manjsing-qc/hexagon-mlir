# SPDX-License-Identifier: BSD-3-Clause
"""Static shape configuration for SmolLM-135M.

Values match HuggingFaceTB/SmolLM-135M and the fp16 ONNX export at
onnx-community/SmolLM-135M-ONNX (``onnx/model_fp16.onnx``).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SmolConfig:
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    head_dim: int = 64
    vocab_size: int = 49152
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    # Hexagon tl.dot / tl.arange tiles. Sequence length must be a power of two
    # so the attention query block fits in one program.
    block_k: int = 64
    block_n: int = 64

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_out(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.num_key_value_heads * self.head_dim
