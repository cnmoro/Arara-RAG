#!/usr/bin/env bash
# Run every arara-rag benchmark suite. Bench-only deps: datasets, pyarrow.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.cache/hf}"
mkdir -p "$HF_HOME"
for suite in core brtaxqa cxm25 sweep; do
  echo "================ suite: $suite ================"
  .venv/bin/python -m bench.run --suite "$suite"
done
echo "================ done ================"
