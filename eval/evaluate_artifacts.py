#!/usr/bin/env python3
"""Evaluate prediction artifacts without importing the reproduced method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_MAP = {
    "roberta_original": "roberta_original",
    "tensorllm_best": "tensorllm_roberta",
}


def score_predictions(path: Path) -> dict:
    total = 0
    correct = 0
    loss = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            correct += int(row["prediction"] == row["label"])
            loss += float(row["nll"])
    if total == 0:
        raise RuntimeError(f"No predictions found in {path}")
    return {"accuracy": 100.0 * correct / total, "loss": loss / total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/tensorllm_fever/manifest.json")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    results = {}
    for condition in manifest["conditions"]:
        mapped = METHOD_MAP.get(condition["name"])
        if mapped is None:
            continue
        metrics = score_predictions(Path(condition["predictions"]))
        metrics["mha_compression_ratio"] = float(condition["compression_ratio"])
        results[mapped] = metrics

    if "roberta_original" not in results or "tensorllm_roberta" not in results:
        raise RuntimeError("Manifest must include roberta_original and tensorllm_best conditions")

    scores = {"experiments": {"fever_roberta_binary": {"results": results}}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
