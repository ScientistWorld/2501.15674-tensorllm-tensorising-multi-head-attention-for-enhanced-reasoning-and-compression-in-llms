#!/bin/bash
# Evaluate the visible train slice from pre-generated prediction artifacts.
#
# Mirrors scripts/evaluate.sh:
#   bash scripts/evaluate_train.sh outputs/tensorllm_fever/manifest.json

set -e

cd "$(dirname "$0")/.."

mkdir -p scoring

MANIFEST="${1:-outputs/tensorllm_fever/manifest.json}"
python eval/train/evaluate.py --manifest "$MANIFEST" --output scoring/scores_train.json
