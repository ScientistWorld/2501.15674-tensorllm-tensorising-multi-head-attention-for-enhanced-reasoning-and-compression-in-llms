#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Torch is CPU-only in the supplied container; use the CPU node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
bash redteam/reference_seed0/run.sh
bash redteam/reference_seed1/run.sh
bash redteam/reference_seed2/run.sh
bash redteam/better_more_hooi/run.sh
bash redteam/weaker_no_tensor/run.sh
bash redteam/weaker_rank_one/run.sh
for scenario in redteam/shortcut_*/run.sh; do
    bash "$scenario"
done
# Save the cheap checks before the broader genuine search.
python redteam/summarize.py
bash redteam/better_visible_search/run.sh
python redteam/summarize.py
