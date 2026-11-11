#!/bin/bash
# Hardening job: build data, reproduce the reference, score it, then build and
# score the better/weaker variants and shortcuts through the unchanged scorers.
# Resumable: finished stages leave a marker in outputs/hardening/stages/.

set -euo pipefail

cd "$(dirname "$0")/../.."

H=outputs/hardening
mkdir -p "$H/stages"
done_stage() { [ -e "$H/stages/$1" ]; }
mark() { touch "$H/stages/$1"; echo "=== stage $1 finished at $(date)"; }

python - <<'PY'
import os, sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "gpus", torch.cuda.device_count(), "cpus", len(os.sched_getaffinity(0)), flush=True)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), flush=True)
PY

# 1. task data (visible split in data/tasks, hidden split in eval/test/data)
if ! done_stage data; then
  python data/prepare_tasks.py --fever-dev data/downloads/raw/fever/paper_dev.jsonl \
    --bios-parquet eval/test/raw/bias_in_bios/dev.parquet
  python eval/test/prepare_test.py --raw-dir eval/test/raw
  mark data
fi

# 2. smoke test of method + both scorers on tiny prefixes
if ! done_stage smoke; then
  rm -rf "$H/smoke"
  python method/tensorllm_roberta.py --layers 10 --r1 128 --r2 32 --r3 3 --limit 64 \
    --out-dir "$H/smoke/submissions/smoke_method" --log "$H/smoke/log.json"
  python eval/train/evaluate.py --submissions "$H/smoke/submissions" --output "$H/smoke/scores_train.json" \
    --key-sources --debug-limit 64
  python eval/test/evaluate.py --submissions "$H/smoke/submissions" --output "$H/smoke/scores_test.json" \
    --key-sources --debug-limit 64
  mark smoke
fi

if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "No CUDA device visible: the full layer/rank search needs a GPU. Stopping after the smoke test." >&2
  exit 3
fi

# 3. reference: the paper's method, selected on the visible split
if ! done_stage reference; then
  python method/tensorllm_roberta.py
  mark reference
fi

# 4. official scores of what is on disk (reference only)
if ! done_stage scores; then
  bash scripts/evaluate_train.sh
  bash scripts/evaluate.sh
  bash scripts/evaluate_test.sh
  mark scores
fi

# 5. variants and shortcut submissions
if ! done_stage variants; then
  python redteam/hardening/variants.py
  mark variants
fi

# 6. score all of them with both scorers (a forged stale score file is planted
#    first; a key listed only in a score file is planted as well)
if ! done_stage hardening_scores; then
  cat > "$H/fake_scores_train.json" <<'JSON'
{"experiments": {"fever_roberta_binary": {"results": {"shortcut_omitted_method": {"accuracy": 100.0}}},
                 "bios_profession_roberta": {"results": {"shortcut_omitted_method": {"accuracy": 100.0}}}}}
JSON
  cat > "$H/scores_test_all.json" <<'JSON'
{"experiments": {"fever_roberta_binary": {"results": {"shortcut_omitted_method": {"accuracy": 100.0, "loss": 0.0, "mha_compression_ratio": 1000000.0}}}}}
JSON
  rm -rf "$H/dumps_test" "$H/dumps_train"
  python eval/test/evaluate.py --submissions "$H/submissions" --output "$H/scores_test_all.json" \
    --key-sources "$H/fake_scores_train.json" --dump-dir "$H/dumps_test"
  python eval/train/evaluate.py --submissions "$H/submissions" --output "$H/scores_train_all.json" \
    --key-sources "$H/fake_scores_train.json" --dump-dir "$H/dumps_train"
  mark hardening_scores
fi

# 7. tampered fixed model (on a symlinked copy; the real files are untouched)
if ! done_stage tamper; then
  rm -rf "$H/submissions_ref_only"
  mkdir -p "$H/submissions_ref_only"
  cp -r outputs/submissions/tensorllm_roberta "$H/submissions_ref_only/"
  python redteam/hardening/tamper_check.py
  mark tamper
fi

# 8. analysis + committed copies of the small result files
python redteam/hardening/analyze.py
R=redteam/hardening/results
mkdir -p "$R"
cp "$H/analysis.json" "$H/scores_test_all.json" "$H/scores_train_all.json" "$H/scores_test_tampered.json" "$R/"
cp "$H/greedy_layers.json" "$H/supervised_ft.json" "$H/laser_mha_log.json" "$H/variants_done.json" "$R/" || true
cp outputs/tensorllm_search/search_log.json "$R/reference_search_log.json"
cp outputs/submissions/tensorllm_roberta/selection.json "$R/reference_selection.json"
for d in "$H"/submissions/*/; do
  name=$(basename "$d")
  for p in "$d"*/patch.json; do
    if [ -e "$p" ]; then
      task=$(basename "$(dirname "$p")")
      mkdir -p "$R/patches/$name/$task"
      cp "$p" "$R/patches/$name/$task/"
    fi
  done
done

python validate.py --post-design || true
echo "=== hardening job finished at $(date)"
