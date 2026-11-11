#!/bin/bash
# Reproduce the reference end to end (inside the compute container, GPU).
#
# Assumes scripts/download.sh ran on an internet-enabled node.  Runs the
# paper's method, then scores it (and the evaluator-owned unmodified model) on
# the visible split and on the held-out test split.

set -e

cd "$(dirname "$0")/.."

if [ ! -e data/downloads/roberta-base/model.safetensors ] || [ ! -e data/tasks/tasks.json ]; then
  echo "Missing model or task data. Run scripts/download.sh on an internet-enabled node first." >&2
  exit 1
fi

bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate.sh
bash scripts/evaluate_test.sh
