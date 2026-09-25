# SPDX-License-Identifier: BSD-3-Clause
"""Download the SmolLM-135M fp16 ONNX model."""

import argparse
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/onnx-community/SmolLM-135M-ONNX/resolve/main/onnx/model_fp16.onnx"
DEFAULT = Path("/home/ubuntu/hexagon-artifacts/models/smollm-135m/model_fp16.onnx")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file() and args.output.stat().st_size > 0:
        print(f"already present: {args.output}")
        return
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
