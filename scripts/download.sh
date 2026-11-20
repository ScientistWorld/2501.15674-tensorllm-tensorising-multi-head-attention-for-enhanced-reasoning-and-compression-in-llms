#!/bin/bash
# All fetched artifacts, from public pinned sources. Visible/test task files
# are tracked; raw sources capable of regenerating hidden labels stay in eval/test.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=e2da8e2f811d1448a5b465c236feacd80ffbac7b
mkdir -p data/downloads/roberta-base
for file in config.json merges.txt vocab.json tokenizer.json tokenizer_config.json model.safetensors; do
  target="data/downloads/roberta-base/${file}"
  if [ ! -s "$target" ]; then
    curl --retry 4 -fL "https://huggingface.co/FacebookAI/roberta-base/resolve/${REV}/${file}" -o "${target}.part"
    mv "${target}.part" "$target"
  fi
done
# The pinned raw-data fetch/rebuild commands live exclusively behind the test boundary.
# They are unnecessary for a fresh clone because the small processed data is tracked.
if [ ! -s eval/test/data/MANIFEST.json ]; then
  python eval/test/prepare_test.py
fi
