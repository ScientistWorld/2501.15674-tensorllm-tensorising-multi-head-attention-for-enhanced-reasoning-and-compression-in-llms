#!/bin/bash
# Download every external artifact this environment needs.
#
# Run on a node with internet access BEFORE submitting compute jobs.  The script
# is idempotent: every fetch is skipped when its target already exists.
#
# Estimated total download size: ~0.52 GB (roberta-base 0.50 GB, FEVER 2 MB,
# Bias in Bios dev split 10 MB, removed again after use).
#
# What is fetched where:
#   data/downloads/roberta-base/   fixed pretrained model (pinned revision)
#   data/downloads/raw/fever/      FEVER v1.0 paper_dev (source of the visible
#                                  FEVER validation split)
# The visible validation splits in data/tasks/ are committed; they are rebuilt
# here only if missing.  The Bias in Bios dev parquet also holds held-out rows,
# so it is fetched to a temporary directory and deleted after the build.
# Held-out test data is built only by eval/test/prepare_test.py.

set -e

cd "$(dirname "$0")/.."

REV=e2da8e2f811d1448a5b465c236feacd80ffbac7b
mkdir -p data/downloads/roberta-base data/downloads/raw/fever

for file in config.json merges.txt vocab.json tokenizer.json tokenizer_config.json model.safetensors; do
  if [ ! -e "data/downloads/roberta-base/${file}" ]; then
    curl -fL "https://huggingface.co/FacebookAI/roberta-base/resolve/${REV}/${file}" \
      -o "data/downloads/roberta-base/${file}"
  fi
done

if [ ! -e data/downloads/raw/fever/paper_dev.jsonl ]; then
  curl -fL https://fever.ai/download/fever/paper_dev.jsonl -o data/downloads/raw/fever/paper_dev.jsonl
fi

if [ ! -e data/tasks/fever_val.jsonl ] || [ ! -e data/tasks/bios_profession_val.jsonl ] || [ ! -e data/tasks/tasks.json ]; then
  tmpdir="$(mktemp -d)"
  curl -fL "https://huggingface.co/datasets/LabHC/bias_in_bios/resolve/main/data/dev-00000-of-00001-e6551072fff26949.parquet" \
    -o "${tmpdir}/dev.parquet"
  python data/prepare_tasks.py --fever-dev data/downloads/raw/fever/paper_dev.jsonl --bios-parquet "${tmpdir}/dev.parquet"
  rm -rf "${tmpdir}"
fi
