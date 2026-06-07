#!/bin/bash
# Score the train/test evaluation split from existing prediction artifacts.

set -euo pipefail

cd /home/user

bash scripts/evaluate_train.sh outputs/tensorllm_fever/manifest.json
bash scripts/evaluate_test.sh outputs/tensorllm_fever/manifest.json
