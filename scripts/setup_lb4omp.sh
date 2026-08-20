#!/usr/bin/env bash
# Clone and build LB4OMP (extended LLVM OpenMP runtime with DLS auto-selection).
# Usage: ./scripts/setup_lb4omp.sh [install-dir]
#
# Produces libomp.so at <install-dir>/build/runtime/src/libomp.so
# Inject via LD_LIBRARY_PATH to replace the system libomp at runtime.

set -euo pipefail

LB4OMP_DIR="${1:-$(pwd)/lb4omp}"

if [ ! -d "$LB4OMP_DIR/.git" ]; then
    echo "Cloning LB4OMP..."
    git clone https://github.com/unibas-dmi-hpc/LB4OMP.git "$LB4OMP_DIR"
fi

echo "Building LB4OMP..."
mkdir -p "$LB4OMP_DIR/build"
cd "$LB4OMP_DIR/build"

cmake \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBOMP_ENABLE_SHARED=ON \
    ..

make -j"$(nproc)"

LIB_PATH="$(pwd)/runtime/src"
if [ -f "$LIB_PATH/libomp.so" ]; then
    echo ""
    echo "Success. LB4OMP library at:"
    echo "  $LIB_PATH/libomp.so"
    echo ""
    echo "Usage:  LD_LIBRARY_PATH=$LIB_PATH ./build/sumfact_harness ..."
else
    echo "ERROR: libomp.so not found at $LIB_PATH" >&2
    exit 1
fi
