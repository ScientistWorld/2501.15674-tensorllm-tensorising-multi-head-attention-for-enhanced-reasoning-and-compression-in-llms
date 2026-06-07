#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   bash scripts/evaluate.sh outputs/tensorllm_fever/manifest.json
#
# Manifest conditions can include a `score_key` field to choose their public
# method name in scoring/scores.json. The original TensorLLM artifacts also
# support the legacy names `roberta_original` and `tensorllm_best`.
#
# Called from run.sh after training, or independently.

set -e

cd /home/user

mkdir -p scoring

MANIFEST="${1:-outputs/tensorllm_fever/manifest.json}"
python eval/evaluate_artifacts.py --manifest "$MANIFEST" --output scoring/scores.json
