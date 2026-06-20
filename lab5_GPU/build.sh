#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

nvcc -std=c++17 -O2 main.cpp train.cpp guessing.cu md5.cpp md5_kernel.cu -o main

echo "Build finished: ./main"