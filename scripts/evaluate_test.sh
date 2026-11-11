#!/bin/bash
# Score every method on disk (outputs/submissions/*), plus every method key
# named in the scoring files, on the held-out test split; writes
# scoring/scores_test.json.  No arguments needed.

set -e

cd "$(dirname "$0")/.."

mkdir -p scoring
python eval/test/evaluate.py "$@"
