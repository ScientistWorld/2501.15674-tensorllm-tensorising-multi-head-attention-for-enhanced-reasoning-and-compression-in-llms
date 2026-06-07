#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0.54 GB
# Estimated disk usage after extraction: 0.55 GB
#
# This script writes all portable artifacts to workspace-local paths so a
# future user with just this repository can reproduce the environment.

set -e

cd /home/user

mkdir -p data/downloads/fever data/downloads/roberta-base data/raw

if [ ! -e data/downloads/fever/shared_task_dev.jsonl ]; then
  curl -L https://fever.ai/download/fever/shared_task_dev.jsonl \
    -o data/downloads/fever/shared_task_dev.jsonl
fi

for file in config.json merges.txt vocab.json tokenizer.json tokenizer_config.json model.safetensors; do
  if [ ! -e "data/downloads/roberta-base/${file}" ]; then
    curl -L "https://huggingface.co/roberta-base/resolve/main/${file}" \
      -o "data/downloads/roberta-base/${file}"
  fi
done

python data/prepare_fever.py \
  --input data/downloads/fever/shared_task_dev.jsonl \
  --output data/fever_binary_dev.jsonl \
  --per-class 160
