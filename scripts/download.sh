#!/bin/bash
# Download every external artifact this environment needs.
#
# Run on a node with internet access BEFORE submitting compute jobs.  The script
# is idempotent: every fetch is skipped when its target already exists.
#
# Estimated total download size: ~0.52 GB (roberta-base 0.50 GB, FEVER 2 MB,
# Bias in Bios dev split 10 MB and BIG-bench qa_wikidata 0.6 MB, both removed
# again after use).
#
# What is fetched where:
#   data/downloads/roberta-base/   fixed pretrained model (pinned revision)
#   data/downloads/raw/fever/      FEVER v1.0 paper_dev (source of the visible
#                                  FEVER validation split)
# The visible validation splits in data/tasks/ are committed; they are rebuilt
# here only if missing.  The Bias in Bios dev parquet and the BIG-bench
# qa_wikidata parquets (tasksource/bigbench, pinned revision) also hold
# held-out rows, so they are fetched to a temporary directory and deleted after
# the build.
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
  python data/prepare_tasks.py --only fever_roberta_binary bios_profession_roberta \
    --fever-dev data/downloads/raw/fever/paper_dev.jsonl --bios-parquet "${tmpdir}/dev.parquet"
  rm -rf "${tmpdir}"
fi

BBQA_REV=b8ee34693560a13a9feac6aa5540449e18b7eddb
if [ ! -e data/tasks/wikidata_qa_val.jsonl ] || ! grep -q wikidata_qa_roberta data/tasks/tasks.json; then
  tmpdir="$(mktemp -d)"
  for split in validation train; do
    curl -fL "https://huggingface.co/datasets/tasksource/bigbench/resolve/${BBQA_REV}/qa_wikidata/${split}-00000-of-00001.parquet" \
      -o "${tmpdir}/${split}.parquet"
  done
  python data/prepare_tasks.py --only wikidata_qa_roberta \
    --bbqa-validation "${tmpdir}/validation.parquet" --bbqa-train "${tmpdir}/train.parquet"
  rm -rf "${tmpdir}"
fi
