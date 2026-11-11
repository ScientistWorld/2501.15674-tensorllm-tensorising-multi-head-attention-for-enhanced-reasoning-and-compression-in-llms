#!/bin/bash
# Run the paper's method (TensorLLM) on every task: search layer and Tucker
# ranks on the visible validation split, then write the selected factor
# patches to outputs/submissions/tensorllm_roberta/.  Needs a GPU for the full
# search (about one hour); extra arguments are passed through.

set -e

cd "$(dirname "$0")/.."

python method/tensorllm_roberta.py "$@"
