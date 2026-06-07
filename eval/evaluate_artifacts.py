#!/usr/bin/env python3
"""Evaluate prediction artifacts without importing the reproduced method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_METHOD_MAP = {
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


def score_key(condition: dict) -> str | None:
    """Return the public score key for a manifest condition.

    Future methods can set ``score_key`` directly in their manifest. The
    fallback map keeps the original TensorLLM artifacts evaluable.
    """

    if "score_key" in condition:
        return str(condition["score_key"])
    return DEFAULT_METHOD_MAP.get(condition["name"])


def resolve_prediction_path(path_text: str, manifest_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return path


def compression_ratio(condition: dict, method_name: str) -> float:
    original = condition.get("original_mha_params")
    compressed = condition.get("compressed_mha_params")
    provided = condition.get("compression_ratio")

    if original is None or compressed is None:
        if provided is None:
            raise RuntimeError(f"{method_name} is missing compression metadata")
        return float(provided)

    if float(compressed) <= 0:
        raise RuntimeError(f"{method_name} has invalid compressed_mha_params={compressed}")
    computed = float(original) / float(compressed)
    if provided is not None and abs(computed - float(provided)) > 1e-6 * max(1.0, computed):
        raise RuntimeError(
            f"{method_name} manifest compression_ratio={provided} does not match "
            f"original_mha_params/compressed_mha_params={computed}"
        )
    return computed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/tensorllm_fever/manifest.json")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    results = {}
    for condition in manifest["conditions"]:
        mapped = score_key(condition)
        if mapped is None:
            continue
        metrics = score_predictions(resolve_prediction_path(condition["predictions"], manifest_path))
        metrics["mha_compression_ratio"] = compression_ratio(condition, mapped)
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
