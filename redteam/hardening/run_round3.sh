#!/bin/bash
# Hardening round 3 (one GPU job): add the BIG-bench WikidataQA experiment.
#   0. build its visible split (data/tasks) and held-out split (eval/test/data)
#      from the pinned parquet files placed beforehand (no internet on compute
#      nodes; the copy under outputs/ is deleted once built);
#   1. self-test the new scoring path (scorers == method-side task);
#   2. reproduce the reference end to end from the final code (FEVER / Bias in
#      Bios search cached, WikidataQA searched now);
#   3. extend every round-1/2 variant and shortcut to the new task, and build the
#      paper's baselines and ablations on all tasks;
#   4. score every submission with both scorers (+ per-example dumps);
#   5. re-run the tampered-model check;
#   6. check that FEVER / Bias in Bios scores of every round-2 submission are
#      unchanged, analyse, and copy the small result files.
# Resumable: finished stages leave a marker in outputs/hardening/stages_round3/.

set -euo pipefail

cd "$(dirname "$0")/../.."

H=outputs/hardening
S="$H/stages_round3"
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

# 0. WikidataQA data (FEVER / Bias in Bios files are left byte-identical)
if ! done_stage data; then
  python data/prepare_tasks.py --only wikidata_qa_roberta \
    --bbqa-validation "$H/bbqa_raw/validation.parquet" --bbqa-train "$H/bbqa_raw/train.parquet"
  python eval/test/prepare_test.py --only wikidata_qa_roberta
  rm -rf "$H/bbqa_raw"
  mark data
fi

# 1. self-test of the new scoring path
if ! done_stage selftest; then
  python redteam/hardening/selftest_round3.py
  mark selftest
fi

# 2. reference end to end from the final code
if ! done_stage reference; then
  bash scripts/reproduce.sh
  mark reference
fi

# 3. variants, baselines, shortcuts on the new task (+ new baselines on all tasks)
if ! done_stage variants; then
  python redteam/hardening/variants_round3.py
  mark variants
fi

# 4. score everything with both scorers; a forged stale score file and a key
#    listed only in a score file are planted first
if ! done_stage scores; then
  cat > "$H/fake_scores_train.json" <<'JSON'
{"experiments": {"fever_roberta_binary": {"results": {"shortcut_omitted_method": {"accuracy": 100.0}}},
                 "bios_profession_roberta": {"results": {"shortcut_omitted_method": {"accuracy": 100.0}}},
                 "wikidata_qa_roberta": {"results": {"shortcut_omitted_method": {"accuracy": 100.0}}}}}
JSON
  cat > "$H/scores_test_all.json" <<'JSON'
{"experiments": {"wikidata_qa_roberta": {"results": {"shortcut_omitted_method": {"accuracy": 100.0, "loss": 0.0, "mha_compression_ratio": 1000000.0}}}}}
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

# 6. regression check against round 2, analysis, committed copies of small results
python - <<'PY'
import json
old = json.load(open("redteam/hardening/results_round2/scores_test_all.json"))["experiments"]
new = json.load(open("outputs/hardening/scores_test_all.json"))["experiments"]
same = differ = 0
for task in ("bios_profession_roberta", "fever_roberta_binary"):
    for key, metrics in old[task]["results"].items():
        if key not in new[task]["results"]:
            continue
        if new[task]["results"][key] == metrics:
            same += 1
        else:
            differ += 1
            print("CHANGED", task, key, metrics, "->", new[task]["results"][key])
print(f"FEVER / Bias in Bios held-out scores unchanged since round 2: {same} identical, {differ} changed", flush=True)
PY
python redteam/hardening/analyze_round3.py
R=redteam/hardening/results_round3
mkdir -p "$R"
cp "$H/analysis_round3.json" "$H/scores_test_all.json" "$H/scores_train_all.json" \
   "$H/scores_test_tampered.json" "$H/compress64_log.json" "$H/laser_all4_log.json" \
   "$H/ablation_single_log.json" "$H/greedy_layers_m05.json" "$H/greedy_layers_margin.json" "$R/"
cp outputs/submissions/tensorllm_roberta/selection.json "$R/reference_selection.json"
cp outputs/tensorllm_search/search_log.json "$R/reference_search_log.json"
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
# a few example generations per method on the held-out WikidataQA split
python - <<'PY'
import json, pathlib
dumps = pathlib.Path("outputs/hardening/dumps_test/wikidata_qa_roberta")
rows = [json.loads(l) for l in open("eval/test/data/wikidata_qa_test.jsonl", encoding="utf-8")]
out = {}
for name in ("roberta_original", "tensorllm_roberta", "better_greedy_layers", "greedy_layers_m05"):
    p = dumps / f"{name}.json"
    if p.exists():
        d = json.loads(p.read_text())
        out[name] = [{"question": rows[i]["question"], "answer": rows[i]["answer"], "generated": d["pred"][i],
                      "correct": d["correct"][i]} for i in range(0, len(rows), 500)]
pathlib.Path("redteam/hardening/results_round3/example_generations.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
PY

python validate.py || true
echo "=== hardening round 3 finished at $(date)"
