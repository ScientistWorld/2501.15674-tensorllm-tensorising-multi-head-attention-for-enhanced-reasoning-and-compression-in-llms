#!/bin/bash
# Validate packaged TensorLLM artifacts and scoring contract.

set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json
bash scripts/evaluate_train.sh outputs/tensorllm_fever/manifest.json
bash scripts/evaluate_test.sh outputs/tensorllm_fever/manifest.json
python validate.py --compare
