#!/usr/bin/env python3
"""Round-3 hardening analysis (pure Python: runs on the login node too).

Inputs (written by redteam/hardening/run_round3.sh):
  outputs/hardening/scores_test_all.json, scores_train_all.json  -- final scorers, every submission
  outputs/hardening/dumps_test/<task>/<method>.json             -- per-example correct / nll / prediction

For every method it reports
  * the score the final scorers wrote, per experiment;
  * illustrative rank deltas against the reference (the system formula:
    experiment weight x sum over metrics of |coefficient| x improvement in the
    better direction / population std of that metric across the cohort) under
    the round-2 gym ("before": FEVER and Bias in Bios, weights 0.5 / 0.5) and
    under the round-3 gym ("after": + WikidataQA, weights 0.2 / 0.4 / 0.4);
  * paired accuracy differences against the reference with their standard
    errors over the same held-out examples.

    python3 redteam/hardening/analyze_round3.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "outputs" / "hardening"
ALL_TASKS = ("bios_profession_roberta", "fever_roberta_binary", "wikidata_qa_roberta")
REF = "tensorllm_roberta"
BASE = "roberta_original"
METRICS = {"accuracy": True, "loss": False, "mha_compression_ratio": True}  # higher_is_better
COEFS = {"accuracy": 1.0, "loss": 0.1, "mha_compression_ratio": 0.25}
GYMS = {
    "before (round-2 gym: bios 0.5, fever 0.5)": {"bios_profession_roberta": 0.5, "fever_roberta_binary": 0.5},
    "after (round-3 gym: bios 0.4, fever 0.2, wikidata 0.4)": {
        "bios_profession_roberta": 0.4, "fever_roberta_binary": 0.2, "wikidata_qa_roberta": 0.4},
}
MATRIX_ABLATIONS = [f"ablation_single_{m}" for m in ("query", "key", "value", "output")]
HONEST = [BASE, REF, "better_greedy_layers", "greedy_layers_m05", "greedy_layers_nomargin", "previous_reference",
          "weaker_low_rank", "weaker_laser_mha", "baseline_laser_mha_all4", *MATRIX_ABLATIONS,
          "variant_head_groups", "variant_qk_vo_split"]
COMPRESSORS = [f"compress64_L{layer}" for layer in range(12)] + ["shortcut_max_compression"]
FAILED_EXAMPLES = ["shortcut_empty", "shortcut_trained_factors", "shortcut_omitted_method", "shortcut_missing_experiment"]


def load(path: Path):
    return json.loads(path.read_text())


def pstd(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def rank_deltas(table: dict, cohort: list[str], weights: dict) -> dict:
    cohort = [k for k in cohort if all(k in table[t] for t in weights)]
    deltas = {k: 0.0 for k in cohort}
    per_task = {t: {k: 0.0 for k in cohort} for t in weights}
    for task, weight in weights.items():
        for metric, higher in METRICS.items():
            values = [table[task][k][metric] for k in cohort]
            sigma = pstd(values)
            if sigma == 0:
                continue
            ref = table[task][REF][metric]
            for k in cohort:
                improvement = (table[task][k][metric] - ref) * (1 if higher else -1)
                contribution = weight * abs(COEFS[metric]) * improvement / sigma
                deltas[k] += contribution
                per_task[task][k] += contribution
    return {"total": {k: round(v, 4) for k, v in deltas.items()},
            "per_experiment": {t: {k: round(v, 4) for k, v in d.items()} for t, d in per_task.items()}}


def paired_noise(dumps: Path, task: str, a: str, b: str):
    """Paired difference of accuracy (points) with its standard error over the same examples."""
    pa, pb = dumps / task / f"{a}.json", dumps / task / f"{b}.json"
    if not (pa.exists() and pb.exists()):
        return None
    ca, cb = load(pa)["correct"], load(pb)["correct"]
    n = len(ca)
    diffs = [x - y for x, y in zip(ca, cb)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return {"diff": round(100 * mean, 3), "se": round(100 * se, 3),
            "ci95": [round(100 * (mean - 1.96 * se), 3), round(100 * (mean + 1.96 * se), 3)],
            "z": round(mean / se, 2) if se > 0 else None,
            "discordant": sum(1 for d in diffs if d != 0)}


def main() -> None:
    test = load(H / "scores_test_all.json")
    train = load(H / "scores_train_all.json")
    tasks = [t for t in ALL_TASKS if t in test["experiments"]]
    table = {task: test["experiments"][task]["results"] for task in tasks}
    train_table = {task: train["experiments"][task]["results"] for task in tasks if task in train["experiments"]}
    methods = sorted(test["metadata"]["methods"])
    present = set(methods)
    cohorts = {
        "honest": [k for k in HONEST if k in present],
        "honest+compressors": [k for k in HONEST + COMPRESSORS if k in present],
        "honest+failed": [k for k in HONEST + FAILED_EXAMPLES if k in present],
        "honest+compressors+failed": [k for k in HONEST + COMPRESSORS + FAILED_EXAMPLES if k in present],
    }
    report = {"scores_test": {}, "scores_train": {}, "gate": {}, "rank": {}, "paired_noise_test": {},
              "paired_noise_train": {}, "cohorts": cohorts, "gates_meta": test["metadata"]["gates"]}
    for task in tasks:
        report["scores_test"][task] = {k: {m: round(v, 4) for m, v in table[task][k].items()} for k in sorted(table[task])}
        if task in train_table:
            report["scores_train"][task] = {k: {m: round(v, 4) for m, v in train_table[task][k].items()}
                                            for k in sorted(train_table[task])}
        report["gate"][task] = {
            k: {kk: test["metadata"]["methods"][k][task].get(kk) for kk in
                ("status", "reasoning_gate_passed", "gate_z_vs_unmodified", "gate_z_vs_majority",
                 "raw_accuracy", "raw_loss", "raw_loss_laser_convention", "raw_compression_ratio",
                 "touched_layers", "reason")}
            for k in methods if task in test["metadata"]["methods"][k]
        }
        report["gate"][task]["_compressors"] = {
            "passed": sum(1 for k in COMPRESSORS if k in present
                          and test["metadata"]["methods"][k][task].get("reasoning_gate_passed")),
            "n": sum(1 for k in COMPRESSORS if k in present),
        }
    for gym, weights in GYMS.items():
        for cohort_name, cohort in cohorts.items():
            report["rank"][f"{gym} | {cohort_name}"] = rank_deltas(table, cohort, weights)
    for split, dumps in (("test", H / "dumps_test"), ("train", H / "dumps_train")):
        for task in tasks:
            noise = {k: paired_noise(dumps, task, k, REF) for k in methods
                     if k != REF and (dumps / task / f"{k}.json").exists()}
            noise[f"{REF}-vs-{BASE}"] = paired_noise(dumps, task, REF, BASE)
            report[f"paired_noise_{split}"][task] = noise
    (H / "analysis_round3.json").write_text(json.dumps(report, indent=1) + "\n")

    for task in tasks:
        gm = test["metadata"]["gates"].get(task, {})
        print(f"\n== {task} (test, n={gm.get('n_examples')}, majority {gm.get('majority_class_accuracy', 0):.2f}) "
              f"compressors passing the gate: {report['gate'][task]['_compressors']}")
        print(f"{'method':30s} {'acc':>7s} {'loss':>7s} {'CR':>6s} {'zU':>7s} {'zM':>7s} {'train acc':>9s}  status")
        for k in methods:
            st = test["metadata"]["methods"][k].get(task, {})
            r = table[task].get(k, {})
            tr = train_table.get(task, {}).get(k, {})
            print(f"{k:30s} {r.get('accuracy', 0):7.2f} {r.get('loss', 0):7.3f} {r.get('mha_compression_ratio', 0):6.2f} "
                  f"{(st.get('gate_z_vs_unmodified') or 0):7.2f} {(st.get('gate_z_vs_majority') or 0):7.2f} "
                  f"{tr.get('accuracy', 0):9.2f}  {st.get('status')} {st.get('reason', '') or ''}")
    print("\n== rank deltas vs reference (test)")
    for label, deltas in report["rank"].items():
        top = sorted(deltas["total"].items(), key=lambda kv: -kv[1])
        print(f"-- {label}")
        print("   " + ", ".join(f"{k}={v:+.3f}" for k, v in top))
    print("\n== paired accuracy difference vs reference (test, points)")
    for task in tasks:
        for k, v in report["paired_noise_test"][task].items():
            if v and not k.startswith("shortcut_") and not k.startswith("compress64"):
                print(f"   {task[:8]} {k:32s} {v}")


if __name__ == "__main__":
    main()
