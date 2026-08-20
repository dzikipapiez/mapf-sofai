#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${1:-python}
tag=$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
pybind_dir=$("$python_bin" -m pybind11 --cmakedir)
native="$repo/src/mapf_anytime/solvers/native"

cmake -S "$native/shortest_paths" -B "$native/shortest_paths/build-pybind-$tag" \
  -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$pybind_dir" \
  -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE="$python_bin"
cmake --build "$native/shortest_paths/build-pybind-$tag" --parallel

cmake -S "$native/lacam" -B "$native/lacam/build-pybind-$tag" \
  -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$pybind_dir" \
  -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE="$python_bin"
cmake --build "$native/lacam/build-pybind-$tag" --parallel

cmake -S "$native/mapf_lns" -B "$native/mapf_lns/build" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$native/mapf_lns/build" --parallel
