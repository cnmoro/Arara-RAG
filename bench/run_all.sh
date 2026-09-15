#!/usr/bin/env bash
# Run every arara-rag benchmark suite.
#
# Uses --suite all so the retrieval suites share one index cache; running the
# suites as separate processes rebuilds each index.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.cache/hf}"
mkdir -p "$HF_HOME"
echo "================ retrieval suites () ================"
.venv/bin/python -m bench.run --suite all
echo "================ reranking () ================"
.venv/bin/python -m bench.rerank
echo "================ done ================"
