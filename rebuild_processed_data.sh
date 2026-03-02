#!/usr/bin/env bash
set -euo pipefail

# One-click rebuild for all files under data/processed.
# Generated files:
# - train_triplets.jsonl
# - val_triplets.jsonl
# - smoke_eval.jsonl
# - demo_corpus.jsonl
# - constraint_benchmark_v1.jsonl
# - retrieval_corpus_v1.jsonl
# - retrieval_benchmark_v1.jsonl

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SEED="${SEED:-42}"
TRAIN_MAX="${TRAIN_MAX:-20000}"
VAL_MAX="${VAL_MAX:-3000}"
NUM_NEGATION="${NUM_NEGATION:-100}"
NUM_EXCLUSION="${NUM_EXCLUSION:-100}"
NUM_NUMERIC="${NUM_NUMERIC:-100}"

echo "==> Project root: $ROOT_DIR"
echo "==> Params: SEED=$SEED TRAIN_MAX=$TRAIN_MAX VAL_MAX=$VAL_MAX NUM_NEGATION=$NUM_NEGATION NUM_EXCLUSION=$NUM_EXCLUSION NUM_NUMERIC=$NUM_NUMERIC"

if [[ ! -d ".venv" ]]; then
  echo "==> Creating virtual environment (.venv)"
  python -m venv .venv
fi

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source ".venv/bin/activate"

echo "==> Installing dependencies from requirements.txt"
pip install -r requirements.txt

echo "==> Step 1/3: Build triplets + smoke_eval + demo_corpus (SNLI source)"
python experiments/build_triplets.py \
  --dataset snli \
  --train-split train \
  --val-split validation \
  --train-max "$TRAIN_MAX" \
  --val-max "$VAL_MAX" \
  --seed "$SEED"

echo "==> Step 2/3: Build synthetic constraint benchmark"
python experiments/build_constraint_benchmark.py \
  --output-file data/processed/constraint_benchmark_v1.jsonl \
  --num-negation "$NUM_NEGATION" \
  --num-exclusion "$NUM_EXCLUSION" \
  --num-numeric "$NUM_NUMERIC" \
  --seed "$SEED"

echo "==> Step 3/3: Build unified retrieval corpus + benchmark"
python experiments/build_retrieval_benchmark.py \
  --source-eval-file data/processed/constraint_benchmark_v1.jsonl \
  --output-corpus-file data/processed/retrieval_corpus_v1.jsonl \
  --output-benchmark-file data/processed/retrieval_benchmark_v1.jsonl

echo "==> Done. Generated files:"
printf " - %s\n" \
  "data/processed/train_triplets.jsonl" \
  "data/processed/val_triplets.jsonl" \
  "data/processed/smoke_eval.jsonl" \
  "data/processed/demo_corpus.jsonl" \
  "data/processed/constraint_benchmark_v1.jsonl" \
  "data/processed/retrieval_corpus_v1.jsonl" \
  "data/processed/retrieval_benchmark_v1.jsonl"

echo "==> Tip: customize with env vars, e.g."
echo "   SEED=123 TRAIN_MAX=5000 VAL_MAX=1000 NUM_NEGATION=50 NUM_EXCLUSION=50 NUM_NUMERIC=50 ./rebuild_processed_data.sh"
