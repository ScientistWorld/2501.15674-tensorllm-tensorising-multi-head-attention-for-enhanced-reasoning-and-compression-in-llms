#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models were downloaded on an internet-enabled node via
#     scripts/download.sh before job submission
#   - Run from any checkout path
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd "$(dirname "$0")/.."

if [ ! -e data/downloads/roberta-base/model.safetensors ] || [ ! -e data/fever_binary_dev.jsonl ]; then
  echo "Missing downloaded data/model artifacts. Run scripts/download.sh on an internet-enabled node before submitting compute jobs." >&2
  exit 1
fi

bash scripts/method.sh
