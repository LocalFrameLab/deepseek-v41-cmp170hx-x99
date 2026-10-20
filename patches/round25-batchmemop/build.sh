#!/usr/bin/env bash
set -euo pipefail
# Build the round-25 BatchMemOp handoff candidate library.
# CUDA_HOME defaults to /usr/local/cuda.
here="$(cd "$(dirname "$0")" && pwd)"
cuda="${CUDA_HOME:-/usr/local/cuda}"

c++ -std=c++17 -O3 -fPIC -shared -pthread \
  -I"$cuda/include" \
  "$here/cuda_memop_moe_v4.cpp" "$here/engram_ssd.cpp" \
  -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" \
  -L"$cuda/lib64/stubs" -lcudart -lcuda \
  -o "$here/libdsv41_cuda_memop.so"

sha256sum "$here/cuda_memop_moe_v4.cpp" "$here/engram_ssd.cpp" "$here/libdsv41_cuda_memop.so"
