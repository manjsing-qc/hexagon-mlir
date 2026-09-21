#!/usr/bin/env bash
# Shared environment for building and running Hexagon-MLIR in a Cloud Agent.
# Source this file to get every path/var (and the activated Python venv) needed
# to build the compiler and run the LIT / Triton / PyTorch tests:
#
#     source /workspace/.cursor/hexagon_env.sh
#
# Heavy, stable artifacts (host clang toolchain, Qualcomm SDK/Tools/HexKL, the
# from-source LLVM build, and the Python venv) live under $BASE_DIR and are
# captured in the environment snapshot, so they are not rebuilt on every boot.

export BASE_DIR=${BASE_DIR:-/home/ubuntu/hexagon-deps}
export HEXAGON_MLIR_ROOT=${HEXAGON_MLIR_ROOT:-/workspace}
export TRITON_ROOT=$HEXAGON_MLIR_ROOT/triton

# Host clang/llvm 13 toolchain. clang-13 is wrapped (in $BASE_DIR/bin) to force
# the gcc-12 toolchain, because Ubuntu 24.04 defaults to gcc-14 which clang-13
# does not support and which has no matching libstdc++.so for -lstdc++.
export HOST_TOOLCHAIN=$BASE_DIR/HOST_TOOLCHAIN
export PATH="$BASE_DIR/bin:$HOST_TOOLCHAIN/bin:$PATH"
export CC="$BASE_DIR/bin/clang"
export CXX="$BASE_DIR/bin/clang++"

# Qualcomm Hexagon components
export HEXAGON_SDK_VERSION=6.4.0.2
export HEXAGON_SDK_ROOT=$BASE_DIR/HEXAGON_SDK/Hexagon_SDK/$HEXAGON_SDK_VERSION
export HEXAGON_TOOLS=$BASE_DIR/HEXAGON_TOOLS/Tools
export HEXKL_ROOT=$BASE_DIR/HEXKL_DIR/hexkl_addon

# LLVM built from source at the Triton-pinned revision
export LLVM_PROJECT_BUILD_DIR=$BASE_DIR/LLVM_DIR/llvm-project/build

# Python virtual environment (Python 3.11)
export CONDA_ENV=$BASE_DIR/mlir-env
if [ -f "$CONDA_ENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_ENV/bin/activate"
fi

# Target Hexagon architecture (v75 by default)
export HEXAGON_ARCH_VERSION=${HEXAGON_ARCH_VERSION:-75}

# Python version + Triton plugin wiring
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo 3.11)
export PYTHON_VERSION
export TRITON_HOME=$HEXAGON_MLIR_ROOT
export TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend"
export TRITON_SHARED_OPT_PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export PATH="$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt:$PATH"
export PYTHONPATH="$TRITON_ROOT/python:${PYTHONPATH:-}"
