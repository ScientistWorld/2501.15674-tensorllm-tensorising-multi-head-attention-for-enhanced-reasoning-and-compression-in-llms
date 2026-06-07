#!/bin/bash
# Evaluate the held-out test slice from pre-generated prediction artifacts.
#
# Mirrors scripts/evaluate.sh:
#   bash scripts/evaluate_test.sh outputs/tensorllm_fever/manifest.json
#
# If scoring/scores_train.json exists, eval/test/evaluate.py scores exactly
# the method keys present in the train slice.

set -e

cd "$(dirname "$0")/.."

mkdir -p scoring

MANIFEST="${1:-outputs/tensorllm_fever/manifest.json}"
python eval/test/evaluate.py \
  --manifest "$MANIFEST" \
  --output scoring/scores_test.json \
  --train-scores scoring/scores_train.json
