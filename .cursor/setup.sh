#!/usr/bin/env bash
#
# Idempotent Cloud Agent bootstrap for Hexagon-MLIR.
#
# The heavy, stable artifacts (host clang-13 toolchain, Qualcomm Hexagon
# SDK/Tools/HexKL, the from-source LLVM build, and the Python 3.11 venv with all
# pip requirements) are baked into the environment snapshot under $BASE_DIR, so
# this script only reconciles the repository-local state after checkout:
#   * initialize/patch the triton + triton_shared submodules, and
#   * build Triton with the Hexagon backend (only if not already built).
#
# It is safe to run repeatedly and terminates once the compiler is ready.
set -euo pipefail

# The Hexagon-MLIR repository is always checked out at /workspace in a Cloud
# Agent. This script is intentionally location-independent: an identical copy is
# baked into the snapshot at $BASE_DIR so install/start can run it regardless of
# which branch is checked out (the committed copy under .cursor/ mirrors it).
REPO_DIR="${HEXAGON_MLIR_ROOT:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/hexagon_env.sh"

echo "==> Hexagon-MLIR setup starting (BASE_DIR=$BASE_DIR)"

# 1. Sanity-check snapshot-provided dependencies.
missing=0
for d in "$HOST_TOOLCHAIN/bin" "$HEXAGON_SDK_ROOT" "$HEXAGON_TOOLS" "$HEXKL_ROOT" "$CONDA_ENV/bin"; do
  if [ ! -e "$d" ]; then echo "WARNING: expected dependency missing: $d"; missing=1; fi
done
[ "$missing" = "0" ] && echo "==> Snapshot dependencies present."

# 2. Ensure triton + triton_shared submodules are initialized and patched.
if [ ! -d "$REPO_DIR/triton" ] || [ ! -d "$REPO_DIR/triton_shared" ]; then
  echo "==> Initializing triton/triton_shared submodules..."
  bash "$REPO_DIR/ci/setup_submodules.sh"
else
  echo "==> Submodules present; ensuring Qualcomm patches are applied..."
  bash "$REPO_DIR/ci/apply_patches.sh"
fi

# 3. Ensure the Triton-compatible LLVM is built (slow fallback; normally present
#    in the snapshot).
if [ ! -f "$LLVM_PROJECT_BUILD_DIR/bin/mlir-opt" ]; then
  echo "==> LLVM not found; building from source (this is slow)..."
  bash "$BASE_DIR/build_llvm.sh"
fi

# 4. Build Triton + Hexagon backend if the backend tool is missing.
BACKEND_BIN="$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt"
if [ ! -f "$BACKEND_BIN" ]; then
  echo "==> Building Triton with the Hexagon backend..."
  # A first clean build can hit a generated-header (.h.inc) ordering race in the
  # backend; a second incremental build resolves it deterministically.
  bash "$REPO_DIR/scripts/build_triton.sh" || bash "$REPO_DIR/scripts/build_triton.sh"
else
  echo "==> Triton Hexagon backend already built."
fi

# 5. The LIT configs use execute_external, removed in lit>=23, so pin lit<23 for
#    test runs (a bare "lit" requirement stays satisfied by this version).
if python -c "import lit,sys; sys.exit(0 if tuple(map(int,lit.__version__.split('.')[:1]))<(23,) else 1)" 2>/dev/null; then
  :
else
  echo "==> Pinning lit<23 for LIT test compatibility..."
  pip install --quiet --ignore-installed --force-reinstall 'lit<23'
fi

echo "==> Setup complete. Compiler tool: $(command -v linalg-hexagon-opt || echo "$BACKEND_BIN")"
echo "==> To work in a shell, run:  source $SCRIPT_DIR/hexagon_env.sh"
