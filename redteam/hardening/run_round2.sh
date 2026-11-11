#!/bin/bash
# Hardening round 2 (one GPU job): self-test the scorer changes, reproduce the
# reference from the final code, build the round-2 variants and shortcuts,
# score every submission (round 1 and round 2) through the unchanged scorers,
# re-run the tampered-model check, and analyse.  Resumable: finished stages
# leave a marker in outputs/hardening/stages_round2/.

set -euo pipefail

cd "$(dirname "$0")/../.."

H=outputs/hardening
S="$H/stages_round2"
mkdir -p "$S"
done_stage() { [ -e "$S/$1" ]; }
mark() { touch "$S/$1"; echo "=== stage $1 finished at $(date)"; }

python - <<'PY'
import os, sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "gpus", torch.cuda.device_count(), "cpus", len(os.sched_getaffinity(0)), flush=True)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), flush=True)
PY

# 1. the round-2 scorers materialise every round-1 patch bit-identically,
#    and the new layout / head-subset fields are validated and counted correctly
if ! done_stage selftest; then
  python redteam/hardening/selftest.py
  mark selftest
fi

# 2. reference end to end from the final code (layer x rank search is cached)
if ! done_stage reference; then
  bash scripts/reproduce.sh
  mark reference
fi

# 3. round-2 variants and shortcuts (round-1 submissions stay in place)
if ! done_stage variants; then
  rm -rf "$H/submissions/tensorllm_roberta"
  cp -r outputs/submissions/tensorllm_roberta "$H/submissions/"
  python redteam/hardening/variants_round2.py
  mark variants
fi

# 4. score everything with both scorers; a forged stale score file and a key
#    listed only in a score file are planted first
if ! done_stage scores; then
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
  mark scores
fi

# 5. tampered fixed model (symlinked copy; the real files are untouched)
if ! done_stage tamper; then
  rm -rf "$H/submissions_ref_only"
  mkdir -p "$H/submissions_ref_only"
  cp -r outputs/submissions/tensorllm_roberta "$H/submissions_ref_only/"
  python redteam/hardening/tamper_check.py
  mark tamper
fi

# 6. analysis + committed copies of the small result files
python redteam/hardening/analyze_round2.py
R=redteam/hardening/results_round2
mkdir -p "$R"
cp "$H/analysis_round2.json" "$H/scores_test_all.json" "$H/scores_train_all.json" \
   "$H/scores_test_tampered.json" "$H/compress64_log.json" "$R/"
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

python validate.py || true
echo "=== hardening round 2 finished at $(date)"
