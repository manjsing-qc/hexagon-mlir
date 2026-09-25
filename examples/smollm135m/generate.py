# SPDX-License-Identifier: BSD-3-Clause
"""Generate text from a SmolLM-135M fp16 ONNX file on Hexagon.

This is the end-to-end entry point. It reads the ONNX initializers, encodes
the prompt, and runs greedy decode through the Triton kernels lowered by
hexagon-mlir.

Example:

    python generate.py \\
        --onnx /path/to/model_fp16.onnx \\
        --prompt "The capital of France is" \\
        --max-new-tokens 8 \\
        --seq-len 16
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onnx_weights import load_onnx_weights
from pipeline import generate

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)
_TOKENIZER_URL = "https://huggingface.co/onnx-community/SmolLM-135M-ONNX/resolve/main/"
_DEFAULT_TOKENIZER = Path("/home/ubuntu/hexagon-artifacts/models/smollm-135m/tokenizer")


def ensure_tokenizer(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    missing = [name for name in _TOKENIZER_FILES if not (directory / name).is_file()]
    if not missing:
        return directory
    import urllib.request

    for name in missing:
        destination = directory / name
        print(f"downloading tokenizer file {name}", flush=True)
        urllib.request.urlretrieve(_TOKENIZER_URL + name, destination)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path, help="SmolLM-135M fp16 ONNX file")
    parser.add_argument("--prompt", required=True, help="Text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16, help="Fixed context length (power of two)")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Decoder layers to run. Default is the full 30-layer model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=_DEFAULT_TOKENIZER,
        help="Directory with the SmolLM tokenizer files",
    )
    args = parser.parse_args()
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX file not found: {args.onnx}")

    tokenizer_dir = ensure_tokenizer(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(f"prompt tokens ({len(prompt_ids)}): {prompt_ids}", flush=True)

    weights = load_onnx_weights(args.onnx, num_layers=args.num_layers)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = 0
    new_ids = generate(
        weights,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        eos_id=int(eos_id),
    )
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    print("completion:", completion, flush=True)
    print("new_token_ids:", new_ids, flush=True)


if __name__ == "__main__":
    main()
