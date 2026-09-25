# SmolLM-135M fp16 on Hexagon

## Scope

SmolLM-135M (`HuggingFaceTB/SmolLM-135M`) is a 30-layer Llama model:

| | |
|---|---|
| hidden | 576 |
| intermediate | 1536 |
| query heads | 9 |
| KV heads | 3 (grouped-query, groups of 3) |
| head dim | 64 |
| vocab | 49152, embeddings tied to the LM head |
| norm | RMSNorm, eps 1e-5 |
| activation | SwiGLU (SiLU) |
| positions | RoPE, theta 10000 |
| weights | fp16 ONNX, about 270MB |

The file used here is `onnx-community/SmolLM-135M-ONNX`, path `onnx/model_fp16.onnx`.

That ONNX graph is a **one-token decode** export: `input_ids` plus `past_key_values.{0..29}.{key,value}` and an attention mask whose length is `past + 1`. It is not a graph this compiler can ingest. hexagon-mlir lowers Triton (and Torch-MLIR) to Linalg, then to Hexagon LLVM. There is no ONNX dialect in the pipeline.

So the pipeline is:

1. Read fp16 initializers out of the ONNX file.
2. Run a static-shape causal **prefill** whose math matches those weights.
3. Lower every math op as a Triton kernel through the existing Hexagon backend.

Host code owns the parts the compiler does not express well: the layer loop, the embedding gather (`[49152, 576]` table, indirect load), and the grouped-query repeat that expands 3 KV heads to 9.

## Kernels

All compute is fp16 in, fp32 accumulate where a reduction or matmul needs it, fp16 out.

| Op | Kernel | Notes |
|---|---|---|
| residual add | `residual_kernel` | flat vector add |
| RMSNorm | `rms_norm_kernel` | variance in fp32 |
| linear | `linear_tile_kernel` | `y = x @ W`, `W` is ONNX `[K, N]`. K is reduced in steps of 64. N is split into power-of-two column tiles (512, 256, 128, 64, …) because 192 / 576 / 1536 are not powers of two |
| RoPE | `rope_kernel` | Llama half-rotation, cos/sin table built on the host |
| SwiGLU | `swiglu_kernel` | `silu(gate) * up` |
| attention | `causal_attention_kernel` | causal FlashAttention-2 forward: online softmax, fp32 row max/sum, fp16 `tl.dot` |

Attention is launched once per head. The existing Hexagon flash-attention test is a single non-causal head with the whole query block resident in one program. This kernel keeps that structure and adds the causal mask inside the key/value loop. Sequence length has to be a power of two (`tl.arange` / `tl.dot`).

## What this does not do

- It does not compile the ONNX graph itself.
- It does not grow a KV cache. Dynamic sequence length is unsupported in this Triton backend, so decode would be a separate static kernel per step.
- It does not run all 30 layers on the simulator as the default check. Each kernel launch is a compile plus a `hexagon-sim` process. A full forward is a long series of those launches. The layer schedule is written so a prefix of layers can be selected.
- The tied LM head (`[seq, 576] @ [576, 49152]`) is implemented with the same linear kernel and is off by default on the simulator because of the output width.
- int4 / bnb ONNX variants are out of scope. The source checkpoint is bfloat16; this path casts the published fp16 export.

## Files

- `config.py` — SmolLM-135M constants
- `onnx_weights.py` — map ONNX initializers, including unnamed MatMul weights, into layer tensors
- `reference.py` — PyTorch prefill used as the numeric reference
- `kernels.py` — Triton kernels and launch helpers
- `pipeline.py` — layer schedule on top of the kernels
- `fetch_model.py` — download `model_fp16.onnx`
- `test_reference.py` — weight shapes, and a 1-token compare against ONNX Runtime
- `test_kernels.py` — hexagon-sim checks

## Run

Download once (the checked-in tree does not contain the 270MB model):

```bash
python examples/smollm135m/fetch_model.py
export SMOLLM_ONNX=/home/ubuntu/hexagon-artifacts/models/smollm-135m/model_fp16.onnx
```

CPU reference against ONNX Runtime, from `examples/smollm135m` with the project virtualenv:

```bash
python test_reference.py
```

Simulator, with the same environment used for `test/python/triton/test_vec_add.py` (`RUN_ON_SIM=1`, `HEXAGON_ARCH_VERSION=75`, Hexagon SDK 6.6 and Tools 19 on `PATH`):

```bash
python test_kernels.py residual
python test_kernels.py linear
python test_kernels.py attention
python test_kernels.py layer
```

`layer` runs one full decoder block at sequence length 16 and the real hidden size, on random fp16 weights, and compares it to `reference.py`.

## What was run

On hexagon-sim v75, fp16, against the PyTorch reference:

| Check | max abs error |
|---|---|
| residual, seq 16 × hidden 576 | 0.00391 |
| RMSNorm, same shape | 0.00049 |
| SwiGLU, seq 16 × intermediate 1536 | 0.00781 |
| linear, 16×64×64 and a 576-wide q_proj | ≤ 0.00006 |
| Llama RoPE, 9 heads × seq 16 × dim 64 | 0 |
| causal FlashAttention-2, seq 32 × dim 64 | 0.00002 |
| one full decoder layer, SmolLM widths, seq 16 | 0.00012 |

The fp16 ONNX checkpoint itself was checked on CPU: a one-token prefill through all 30 layers matches ONNX Runtime on the published decode graph (empty KV cache). The top token agrees, and the max logit difference is 0.16. That graph is a decode export, so the length-1 prefill is the case where the two schedules are the same math.
