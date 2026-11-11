#!/usr/bin/env python3
"""Summarise the hardening runs: scores, paired test noise, illustrative rank.

Reads the scorer outputs written by ``redteam/hardening/run.sh`` and writes
``outputs/hardening/analysis.json`` (copied to ``redteam/hardening/results/``).

* Paired noise: every method is scored on the same test prompts, so the
  standard error of an accuracy difference is estimated with a paired
  bootstrap over test examples (2,000 resamples, fixed seed).
* Illustrative rank: sum over experiments (weight) and metrics (|coefficient|)
  of the improvement over the reference in the metric's better direction,
  divided by the metric's standard deviation across the honest cohort.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "hardening"
REFERENCE = "tensorllm_roberta"
BASELINE = "roberta_original"
HONEST = ["roberta_original", "tensorllm_roberta", "previous_reference", "weaker_low_rank",
          "weaker_laser_mha", "better_greedy_layers", "greedy_layers_nomargin"]


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def paired_bootstrap(a: list[int], b: list[int], n_boot: int = 2000, seed: int = 0) -> dict:
    import numpy as np

    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diff)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for start in range(0, n_boot, 200):
        idx = rng.integers(0, n, size=(min(200, n_boot - start), n))
        boots[start : start + idx.shape[0]] = 100.0 * diff[idx].mean(axis=1)
    return {
        "diff_pp": float(100.0 * diff.mean()),
        "se_pp": float(boots.std()),
        "ci95_pp": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n": int(n),
        "disagreements": int((diff != 0).sum()),
    }


def rank_deltas(scores: dict, reference: dict, cohort: list[str]) -> dict:
    exps = reference["experiments"]
    out = {}
    for method in scores["experiments"][next(iter(exps))]["results"]:
        total = 0.0
        parts = {}
        for exp_name, exp in exps.items():
            results = scores["experiments"][exp_name]["results"]
            if method not in results or REFERENCE not in results:
                continue
            for metric, spec in exp["metrics"].items():
                values = [results[m][metric] for m in cohort if m in results]
                sigma = statistics.pstdev(values) if len(values) > 1 else 0.0
                if sigma <= 0:
                    continue
                sign = 1.0 if spec["higher_is_better"] else -1.0
                delta = sign * (results[method][metric] - results[REFERENCE][metric]) / sigma
                contrib = exp["weight"] * abs(spec["coefficient"]) * delta
                parts[f"{exp_name}.{metric}"] = round(contrib, 4)
                total += contrib
        out[method] = {"delta": round(total, 4), "parts": parts}
    return out


def main() -> int:
    reference = json.loads((ROOT / "scoring" / "reference.json").read_text())
    test = load(OUT / "scores_test_all.json")
    train = load(OUT / "scores_train_all.json")
    if test is None:
        print("missing scores_test_all.json", file=sys.stderr)
        return 1
    analysis = {"test": test["experiments"], "train": train["experiments"] if train else None,
                "gates_test": test["metadata"].get("gates"), "methods_test": test["metadata"]["methods"]}

    noise = {}
    dump_root = OUT / "dumps_test"
    for exp_dir in sorted(p for p in dump_root.iterdir() if p.is_dir()) if dump_root.exists() else []:
        dumps = {p.stem: json.loads(p.read_text()) for p in exp_dir.glob("*.json")}
        if REFERENCE not in dumps:
            continue
        ref = dumps[REFERENCE]["correct"]
        entry = {}
        for method, dump in sorted(dumps.items()):
            if method == REFERENCE:
                continue
            entry[f"{method}_minus_reference"] = paired_bootstrap(dump["correct"], ref)
        if BASELINE in dumps:
            entry["reference_minus_baseline"] = paired_bootstrap(ref, dumps[BASELINE]["correct"])
            if exp_dir.name == "fever_roberta_binary":
                k = len(ref) - 2617  # the paper's last-20% portion of the ordered FEVER dataset
                entry["reference_minus_baseline_paper_last20pct"] = paired_bootstrap(
                    ref[k:], dumps[BASELINE]["correct"][k:])
                if "better_greedy_layers" in dumps:
                    entry["better_greedy_layers_minus_reference_paper_last20pct"] = paired_bootstrap(
                        dumps["better_greedy_layers"]["correct"][k:], ref[k:])
        noise[exp_dir.name] = entry
    analysis["paired_noise_test"] = noise

    cohort = [m for m in HONEST if m in test["experiments"][next(iter(test["experiments"]))]["results"]]
    analysis["rank_cohort"] = cohort
    analysis["rank_test"] = rank_deltas(test, reference, cohort)
    if train:
        analysis["rank_train"] = rank_deltas(train, reference, cohort)
    for extra in ("scores_test_tampered.json", "scores_test_stale.json"):
        data = load(OUT / extra)
        if data:
            analysis[extra] = {"experiments": data["experiments"],
                               "methods": data["metadata"]["methods"],
                               "fixed_model_verified": data["metadata"]["fixed_model_verified"]}
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    for exp_name, exp in test["experiments"].items():
        print(f"\n## {exp_name}")
        print("| method | accuracy | loss | CR | gate | rank delta (test) |")
        print("|---|---:|---:|---:|---|---:|")
        for method, values in exp["results"].items():
            meta = test["metadata"]["methods"].get(method, {}).get(exp_name, {})
            gate = meta.get("reasoning_gate_passed", meta.get("reason", ""))
            delta = analysis["rank_test"].get(method, {}).get("delta", "")
            print(f"| {method} | {values['accuracy']:.3f} | {values['loss']:.4f} | "
                  f"{values['mha_compression_ratio']:.3f} | {gate} | {delta} |")
    print(json.dumps(noise, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
