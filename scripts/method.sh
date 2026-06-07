#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python method/tensorllm_roberta.py \
  --model-path data/downloads/roberta-base \
  --data data/fever_binary_dev.jsonl \
  --output-dir outputs/tensorllm_fever \
  --limit "${TENSORLLM_LIMIT:-160}" \
  --batch-size "${TENSORLLM_BATCH_SIZE:-8}" \
  --layers "${TENSORLLM_LAYERS:-0,3,6,9,11}" \
  --ranks ${TENSORLLM_RANKS:-384,32,3 256,32,3 192,24,3}

bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json
