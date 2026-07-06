#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_PATH="${1:-./guessdata/Rockyou-singleLined-full.txt}"
GENERATE_LIMIT="${2:-10000000}"
GPU_THRESHOLD="${3:-4096}"
BATCH_FLUSH_SIZE="${4:-131072}"

./main "$DATA_PATH" "$GENERATE_LIMIT" "$GPU_THRESHOLD" "$BATCH_FLUSH_SIZE"