#!/bin/bash
# Smoke-test TensorLLM end-to-end inside the managed container.

set -euo pipefail

cd "$(dirname "$0")/.."

rm -rf outputs/tensorllm_fever_smoke

TENSORLLM_LIMIT=40 \
TENSORLLM_BATCH_SIZE=8 \
TENSORLLM_LAYERS=6 \
TENSORLLM_RANKS="384,48,4" \
TENSORLLM_OUTPUT_DIR=outputs/tensorllm_fever_smoke \
bash scripts/reproduce.sh

bash scripts/evaluate_train.sh outputs/tensorllm_fever_smoke/manifest.json
bash scripts/evaluate_test.sh outputs/tensorllm_fever_smoke/manifest.json

bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json
bash scripts/evaluate_train.sh outputs/tensorllm_fever/manifest.json
bash scripts/evaluate_test.sh outputs/tensorllm_fever/manifest.json
python validate.py --compare
