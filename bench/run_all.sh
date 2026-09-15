#!/usr/bin/env bash
# Run every arara-rag suite for one dense encoder backend.
#   ./bench/run_all.sh [static|use]
#
# Uses --suite all so the retrieval suites share one index cache; running the
# suites as separate processes rebuilds each index.
set -euo pipefail
cd "$(dirname "$0")/.."
ENCODER="${1:-static}"
export HF_HOME="${HF_HOME:-$PWD/.cache/hf}"
mkdir -p "$HF_HOME"
echo "================ retrieval suites (encoder=$ENCODER) ================"
.venv/bin/python -m bench.run --suite all --encoder "$ENCODER"
echo "================ reranking (encoder=$ENCODER) ================"
.venv/bin/python -m bench.rerank --encoder "$ENCODER"
echo "================ done ================"
