#!/bin/bash
# Score every submission on the public validation split -> scoring/scores.json.
#
# A method is a directory outputs/submissions/<method>/ with one factor patch
# per experiment (<experiment>/patch.json + tensors.safetensors); see
# briefing/evaluation.md.  The unmodified roberta-base ("roberta_original") is
# always scored by the evaluator itself.  No arguments needed.

set -e

cd "$(dirname "$0")/.."

mkdir -p scoring
python eval/train/evaluate.py --output scoring/scores.json "$@"
