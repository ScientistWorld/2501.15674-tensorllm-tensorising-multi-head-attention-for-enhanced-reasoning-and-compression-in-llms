#!/bin/bash
# Baseline: the unmodified pretrained roberta-base ("roberta_original").
#
# It needs no artifacts: both evaluators run the fixed model themselves and
# report it next to every submitted method, so scoring is the whole baseline run.

set -e

cd "$(dirname "$0")/.."

bash scripts/evaluate_train.sh
