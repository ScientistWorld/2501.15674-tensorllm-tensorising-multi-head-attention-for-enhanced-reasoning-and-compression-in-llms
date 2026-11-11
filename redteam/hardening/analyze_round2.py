#!/usr/bin/env python3
"""Round-2 hardening analysis (pure Python: runs on the login node too).

Inputs (written by redteam/hardening/run_round2.sh):
  outputs/hardening/scores_test_all.json, scores_train_all.json  -- round-2 scorers, every submission
  outputs/hardening/dumps_test/<task>/<method>.json             -- per-example correct / nll

For every method it reports
  * the round-2 score (what the scorers wrote);
  * the round-1 score, recomputed from the same raw outputs with the round-1 gate
    (credit when accuracy > max(unmodified, majority); otherwise loss floored at
    the unmodified model's loss and compression 1) -- materialisation is
    bit-identical between the rounds (selftest.py), so this is exactly what the
    round-1 scorer would have written;
  * illustrative rank deltas against the reference for several coefficient
    settings and cohorts (the system formula: experiment weight x sum over
    metrics of |coefficient| x improvement in the better direction / population
    std of that metric across the cohort);
  * paired bootstrap noise of accuracy differences against the reference.

    python3 redteam/hardening/analyze_round2.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "outputs" / "hardening"
TASKS = ("bios_profession_roberta", "fever_roberta_binary")
REF = "tensorllm_roberta"
BASE = "roberta_original"
LOSS_CAP, CR_CAP = 30.0, 64.0
METRICS = {"accuracy": True, "loss": False, "mha_compression_ratio": True}  # higher_is_better
COEFS = {
    "round1 (acc 1, loss 0.1, CR 0.25)": {"accuracy": 1.0, "loss": 0.1, "mha_compression_ratio": 0.25},
    "round2 (acc 1, loss 0.1, CR 0.1)": {"accuracy": 1.0, "loss": 0.1, "mha_compression_ratio": 0.1},
}
WEIGHTS = {"bios_profession_roberta": 0.5, "fever_roberta_binary": 0.5}

HONEST = [BASE, REF, "better_greedy_layers", "greedy_layers_nomargin", "previous_reference", "weaker_low_rank",
          "weaker_laser_mha", "variant_head_groups", "variant_qk_vo_split"]
COMPRESSORS = [f"compress64_L{layer}" for layer in range(12)] + ["shortcut_max_compression"]
FAILED_EXAMPLES = ["shortcut_empty", "shortcut_trained_factors", "shortcut_omitted_method"]


def load(path: Path):
    return json.loads(path.read_text())


def round1_scores(scores: dict) -> dict:
    """Round-1 gate applied to the raw outputs recorded by the round-2 scorer."""
    out = {}
    gates = scores["metadata"]["gates"]
    for task in TASKS:
        out[task] = {}
        gate = gates[task]
        threshold = max(gate["baseline_accuracy"], gate["majority_class_accuracy"])
        for key, per_task in scores["metadata"]["methods"].items():
            st = per_task.get(task, {})
            if st.get("status") != "ok":
                out[task][key] = {"accuracy": 0.0, "loss": LOSS_CAP, "mha_compression_ratio": 1.0}
                continue
            if key == BASE:
                out[task][key] = {"accuracy": st["raw_accuracy"], "loss": st["raw_loss"], "mha_compression_ratio": 1.0}
                continue
            passed = st["raw_accuracy"] > threshold + 1e-9
            out[task][key] = {
                "accuracy": st["raw_accuracy"],
                "loss": st["raw_loss"] if passed else max(st["raw_loss"], gate["baseline_loss"]),
                "mha_compression_ratio": min(st["raw_compression_ratio"], CR_CAP) if passed else 1.0,
            }
    return out


def round2_scores(scores: dict) -> dict:
    return {task: scores["experiments"][task]["results"] for task in TASKS}


def pstd(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def rank_deltas(table: dict, cohort: list[str], coefs: dict) -> dict:
    cohort = [k for k in cohort if all(k in table[t] for t in TASKS)]
    deltas = {k: 0.0 for k in cohort}
    for task in TASKS:
        for metric, higher in METRICS.items():
            values = [table[task][k][metric] for k in cohort]
            sigma = pstd(values)
            if sigma == 0:
                continue
            ref = table[task][REF][metric]
            for k in cohort:
                improvement = (table[task][k][metric] - ref) * (1 if higher else -1)
                deltas[k] += WEIGHTS[task] * abs(coefs[metric]) * improvement / sigma
    return {k: round(v, 4) for k, v in deltas.items()}


def paired_noise(dumps: Path, task: str, a: str, b: str):
    """Paired difference of accuracy (points) with its standard error over examples (same prompts)."""
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
            "discordant": sum(1 for d in diffs if d != 0)}


def main() -> None:
    test = load(H / "scores_test_all.json")
    train = load(H / "scores_train_all.json")
    tables = {
        "test_round1": round1_scores(test), "test_round2": round2_scores(test),
        "train_round1": round1_scores(train), "train_round2": round2_scores(train),
    }
    methods = sorted(test["metadata"]["methods"])
    present = set(methods)
    cohorts = {
        "honest": [k for k in HONEST if k in present],
        "honest+compressors": [k for k in HONEST + COMPRESSORS if k in present],
        "honest+compressors+failed": [k for k in HONEST + COMPRESSORS + FAILED_EXAMPLES if k in present],
        "honest+failed": [k for k in HONEST + FAILED_EXAMPLES if k in present],
    }
    report = {"scores": {}, "gate": {}, "rank": {}, "paired_noise_test": {}, "cohorts": cohorts}
    for name, table in tables.items():
        report["scores"][name] = {task: {k: {m: round(v, 4) for m, v in table[task][k].items()}
                                         for k in sorted(table[task])} for task in TASKS}
    for task in TASKS:
        report["gate"][task] = {
            k: {kk: test["metadata"]["methods"][k][task].get(kk) for kk in
                ("status", "reasoning_gate_passed", "gate_z_vs_unmodified", "gate_z_vs_majority",
                 "raw_accuracy", "raw_loss", "raw_compression_ratio", "reason")}
            for k in methods if task in test["metadata"]["methods"][k]
        }
        threshold = max(test["metadata"]["gates"][task]["baseline_accuracy"],
                        test["metadata"]["gates"][task]["majority_class_accuracy"])
        report["gate"][task]["_compressors"] = {
            "round1_passed": sum(1 for k in COMPRESSORS if k in present
                                 and test["metadata"]["methods"][k][task].get("raw_accuracy", 0) > threshold + 1e-9),
            "round2_passed": sum(1 for k in COMPRESSORS if k in present
                                 and test["metadata"]["methods"][k][task].get("reasoning_gate_passed")),
            "n": sum(1 for k in COMPRESSORS if k in present),
        }
    for scoring in ("round1", "round2"):
        for coef_name, coefs in COEFS.items():
            for cohort_name, cohort in cohorts.items():
                label = f"{scoring} scoring | {coef_name} | {cohort_name}"
                report["rank"][label] = rank_deltas(tables[f"test_{scoring}"], cohort, coefs)
    dumps = H / "dumps_test"
    for task in TASKS:
        report["paired_noise_test"][task] = {
            k: paired_noise(dumps, task, k, REF) for k in methods if k != REF and (dumps / task / f"{k}.json").exists()
        }
        report["paired_noise_test"][task][f"{REF}-vs-{BASE}"] = paired_noise(dumps, task, REF, BASE)
    (H / "analysis_round2.json").write_text(json.dumps(report, indent=1) + "\n")

    # short human-readable summary
    for task in TASKS:
        print(f"\n== {task} (test)  gate for compressors: {report['gate'][task]['_compressors']}")
        print(f"{'method':32s} {'raw acc':>8s} {'raw CR':>7s} {'zU':>6s} {'zM':>6s}  round1 (acc/loss/CR)   round2 (acc/loss/CR)")
        for k in methods:
            st = test["metadata"]["methods"][k].get(task, {})
            r1 = tables["test_round1"][task].get(k)
            r2 = tables["test_round2"][task].get(k)
            fmt = lambda r: f"{r['accuracy']:6.2f}/{r['loss']:6.3f}/{r['mha_compression_ratio']:6.2f}"
            print(f"{k:32s} {st.get('raw_accuracy', 0):8.2f} {st.get('raw_compression_ratio', 0):7.2f} "
                  f"{(st.get('gate_z_vs_unmodified') or 0):6.2f} {(st.get('gate_z_vs_majority') or 0):6.2f}  "
                  f"{fmt(r1)}  {fmt(r2)}")
    print("\n== rank deltas vs reference (test)")
    for label, deltas in report["rank"].items():
        top = sorted(deltas.items(), key=lambda kv: -kv[1])
        print(f"-- {label}")
        print("   " + ", ".join(f"{k}={v:+.3f}" for k, v in top[:8]))
    print("\n== paired noise vs reference (test accuracy points)")
    for task in TASKS:
        for k, v in report["paired_noise_test"][task].items():
            if v and not k.startswith("shortcut_") and not k.startswith("compress64"):
                print(f"   {task[:5]} {k:32s} {v}")


if __name__ == "__main__":
    main()
