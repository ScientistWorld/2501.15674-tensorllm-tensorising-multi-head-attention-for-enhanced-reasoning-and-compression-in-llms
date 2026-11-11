#!/bin/bash
# Score every method on disk (outputs/submissions/*) on the public validation
# split; writes scoring/scores_train.json.  No arguments needed.

set -e

cd "$(dirname "$0")/.."

mkdir -p scoring
python eval/train/evaluate.py "$@"
